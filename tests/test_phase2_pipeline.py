"""Phase 2 wiring: history files, Gate 2 driver + report, gate2-apply, and the live strategy runner."""

from __future__ import annotations

import json
import lzma
import struct
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D

import httpx
import pandas as pd
import pytest
import respx
from conftest import FakeAdapter
from sqlalchemy import select
from test_strategies_backtest import synthetic_h1

from tradeengine.backtest import parse_params_id, render_report, run_gate2, save_report
from tradeengine.cli import main as cli_main
from tradeengine.config import InfraConfig
from tradeengine.db import PositionMetaRow, PositionRow
from tradeengine.engine import Engine
from tradeengine.history import (
    bars_to_frame,
    download_dukascopy,
    frame_to_bars,
    history_path,
    load_h1,
    resample_frame,
    write_bars,
)
from tradeengine.models import DEFAULT_INSTRUMENTS, MarketSpec, Quote, Side, Venue
from tradeengine.runner import StrategyRunner
from tradeengine.settings_store import RuntimeSettings
from tradeengine.strategies import Signal, build_frame


# ---------------------------------------------------------------- history files
def test_frames_round_trip_and_resample(tmp_path):
    m1 = synthetic_h1(n=180, seed=1)
    m1.index = pd.date_range("2024-03-01", periods=180, freq="min", tz="UTC")
    h1 = resample_frame(m1, "1h")
    assert len(h1) == 3 and h1["open"].iloc[0] == m1["open"].iloc[0] and h1["high"].iloc[1] == m1["high"].iloc[60:120].max()
    p = history_path(tmp_path, "capital", "EUR/USD", "H1")
    write_bars(p, frame_to_bars(h1, "EUR/USD", "H1"))
    back, note = load_h1(tmp_path, "EUR/USD", crypto=False)
    assert back.index.equals(h1.index) and back["close"].round(8).equals(h1["close"].round(8))
    assert "capital H1, 3 bars" in note
    assert bars_to_frame(frame_to_bars(h1, "X", "H1"))["close"].round(8).equals(h1["close"].round(8))


def test_load_h1_prefers_the_longer_history(tmp_path):
    long = synthetic_h1(n=500)
    short = synthetic_h1(n=100)
    write_bars(history_path(tmp_path, "capital", "EUR/USD", "H1"), frame_to_bars(short, "EUR/USD", "H1"))
    write_bars(history_path(tmp_path, "dukascopy", "EUR/USD", "H1"), frame_to_bars(long, "EUR/USD", "H1"))
    df, note = load_h1(tmp_path, "EUR/USD", crypto=False)
    assert len(df) == 500 and note.startswith("EUR/USD: dukascopy")


@respx.mock
async def test_dukascopy_download_parallel_with_missing_days():
    rec = struct.pack(">IIIIIf", 0, 110000, 110020, 109990, 110030, 1.0)
    respx.get(url__regex=r".*/EURUSD/2024/00/1[56]/BID_candles_min_1\.bi5").mock(return_value=httpx.Response(200, content=lzma.compress(rec, format=lzma.FORMAT_ALONE)))
    respx.get(url__regex=r".*/EURUSD/2024/00/17/BID_candles_min_1\.bi5").mock(return_value=httpx.Response(404))
    async with httpx.AsyncClient() as client:
        bars = await download_dukascopy(client, "EUR/USD", date(2024, 1, 15), date(2024, 1, 18))
    assert [b.open_time.day for b in bars] == [15, 16]


@respx.mock
async def test_capital_history_skips_unavailable_windows():
    from tradeengine.brokers.capital import DEMO_REST, CapitalComAdapter

    respx.post(f"{DEMO_REST}/api/v1/session").mock(return_value=httpx.Response(200, json={"currentAccountId": "A"}, headers={"CST": "c", "X-SECURITY-TOKEN": "x"}))
    ok = {"prices": [{"snapshotTimeUTC": "2026-01-20T00:00:00", "openPrice": {"bid": 1, "ask": 1}, "highPrice": {"bid": 1, "ask": 1},
                      "lowPrice": {"bid": 1, "ask": 1}, "closePrice": {"bid": 1, "ask": 1}}]}
    respx.get(f"{DEMO_REST}/api/v1/prices/EURUSD").mock(side_effect=[httpx.Response(400, json={"errorCode": "error.prices.not-found"}), httpx.Response(200, json=ok)])
    a = CapitalComAdapter("k", "i", "p", list(DEFAULT_INSTRUMENTS.values()), client=httpx.AsyncClient(base_url=DEMO_REST))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    bars = await a.history(DEFAULT_INSTRUMENTS["EUR/USD"], "H1", start, start + timedelta(hours=1500))
    assert len(bars) == 1 and len(a.history_gaps) == 1


# ---------------------------------------------------------------- Gate 2 driver + apply
def test_run_gate2_end_to_end_on_files(tmp_path):
    fx = synthetic_h1(n=24 * 30 * 18, seed=5)
    btc = synthetic_h1(n=24 * 30 * 18, seed=9) * 50000
    write_bars(history_path(tmp_path, "capital", "EUR/USD", "H1"), frame_to_bars(fx, "EUR/USD", "H1"))
    write_bars(history_path(tmp_path, "binance", "BTC/USDT", "H1"), frame_to_bars(btc, "BTC/USDT", "H1"))
    s = RuntimeSettings(instruments=["EUR/USD", "BTC/USDT", "GBP/USD"], walk_forward_train_months=6, walk_forward_test_months=2)
    verdicts, results, rules, notes = run_gate2(tmp_path, s, DEFAULT_INSTRUMENTS, progress=lambda *_: None)
    assert [v.strategy for v in verdicts] == ["trend_pullback", "mean_reversion", "breakout"]
    assert any("GBP/USD: no history" in n for n in notes)
    assert all(not v.passed for v in verdicts)  # synthetic data: 200 trades are out of reach
    btc_trades = [t for r in results["breakout"] if r.symbol == "BTC/USDT" for t in r.oos_trades]
    assert all(t.side == "BUY" for t in btc_trades)  # spot is long-only in the backtest too
    md = render_report(verdicts, results, rules, notes)
    assert "## Verdict" in md and "trend_pullback" in md and "EUR/USD" in md
    out = save_report(tmp_path / "rep", verdicts, results, md)
    assert (out / "report.md").exists() and json.loads((out / "result.json").read_text())["verdicts"]


def test_parse_params_id():
    assert parse_params_id("bb_std=2.5,max_bars=24,rsi_low=30") == {"bb_std": 2.5, "max_bars": 24, "rsi_low": 30}


def test_gate2_apply_enables_only_passed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    for k in ("CAPITAL_API_KEY", "BINANCE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    result = {"verdicts": [
        {"strategy": "trend_pullback", "passed": True, "enabled_symbols": ["EUR/USD"], "latest_params": {"EUR/USD": "max_bars=72,rsi_pull=40,stop_atr=1.5,target_r=3.0,trail_atr=2.0"}},
        {"strategy": "breakout", "passed": False, "enabled_symbols": [], "latest_params": {}},
    ]}
    (tmp_path / "r.json").write_text(json.dumps(result))
    cli_main(["gate2-apply", str(tmp_path / "r.json")])
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path / "data")
    import asyncio

    async def read():
        e = Engine(cfg, adapters={})
        s = await e.settings_store.load()
        await e.stop()
        return s

    s = asyncio.run(read())
    assert s.strategies_enabled == {"trend_pullback": ["EUR/USD"]}
    assert s.strategy_params["trend_pullback"]["EUR/USD"]["target_r"] == 3.0


# ---------------------------------------------------------------- live runner
class HistoryFake(FakeAdapter):
    def __init__(self, h1: pd.DataFrame):
        super().__init__(Venue.BINANCE, "USDT")
        self.h1 = h1

    async def market_spec(self, inst):
        return MarketSpec(D("0.00001"), D("0.00001"), D("0.01"), D("5"))

    async def quote(self, inst):
        last = float(self.h1["close"].iloc[-1])
        return Quote(inst.symbol, D(repr(last)) - D("1"), D(repr(last)) + D("1"), datetime.now(UTC))

    async def history(self, inst, tf, start, end):
        df = self.h1[(self.h1.index >= pd.Timestamp(start)) & (self.h1.index < pd.Timestamp(end))]
        return frame_to_bars(df, inst.symbol, tf)


@dataclass
class LastBarSignal:
    name: str = "scripted"
    default_params: dict = None
    fire: bool = True
    max_bars: int = 5

    def __post_init__(self):
        self.default_params = {}

    def prepare(self, h1, params):
        return build_frame(h1)

    def signal_at(self, cols, i, p, allow_short):
        if not self.fire:
            return None
        c, atr = float(cols["close"][i]), float(cols["atr"][i])
        return Signal(Side.BUY, c, c - 2 * atr, c + 6 * atr, atr, "test", trail_atr=1.0, breakeven_at_r=1.0, max_bars=self.max_bars)


@pytest.fixture
async def runner_env(tmp_path):
    h1 = synthetic_h1(n=24 * 45, seed=4) * 60000
    now = h1.index[-1].to_pydatetime() + timedelta(hours=1, seconds=30)
    fake = HistoryFake(h1)
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'r.db'}", data_dir=tmp_path)
    e = Engine(cfg, adapters={Venue.BINANCE: fake})
    await e.start()
    yield e, fake, h1, now
    await e.stop()


async def test_runner_idle_until_a_strategy_is_enabled(runner_env):
    e, fake, h1, now = runner_env
    strat = LastBarSignal()
    assert await StrategyRunner(e, {"scripted": strat}).on_hour(now) == []
    assert fake.open_calls == 0


async def test_runner_trades_once_per_signal_and_records_management(runner_env):
    e, fake, h1, now = runner_env
    await e.settings_store.update({"strategies_enabled": {"scripted": ["BTC/USDT"]}, "instruments": ["BTC/USDT"]}, "t")
    runner = StrategyRunner(e, {"scripted": LastBarSignal()})
    actions = await runner.on_hour(now)
    assert actions == ["opened BUY BTC/USDT (scripted)"] and fake.open_calls == 1
    async with e.db.sessions() as s:
        meta = (await s.execute(select(PositionMetaRow))).scalar_one()
    assert meta.strategy == "scripted" and meta.max_bars == 5
    assert pd.Timestamp(meta.signal_bar).tz_localize(None) == h1.index[-1].tz_localize(None)
    again = await runner.on_hour(now)  # same hour processed twice (restart)
    assert fake.open_calls == 1 and again == []  # the strategy already holds it: no new signal at all, as in the backtest


async def test_runner_trails_then_time_exits(runner_env):
    e, fake, h1, now = runner_env
    await e.settings_store.update({"strategies_enabled": {"scripted": ["BTC/USDT"]}, "instruments": ["BTC/USDT"]}, "t")
    strat = LastBarSignal()
    runner = StrategyRunner(e, {"scripted": strat})
    await runner.on_hour(now)
    strat.fire = False
    async with e.db.sessions() as s:
        pos = (await s.execute(select(PositionRow))).scalar_one()
    stop0, entry = float(pos.stop), float(pos.entry_price)
    # Two strong up bars: the trade is well past +1R, so the stop must move up.
    t0 = h1.index[-1]
    up = pd.DataFrame(
        {"open": [entry, entry * 1.03], "high": [entry * 1.03, entry * 1.06], "low": [entry, entry * 1.03], "close": [entry * 1.03, entry * 1.06], "volume": [1, 1]},
        index=[t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
    )
    fake.h1 = pd.concat([h1, up])
    actions = await runner.on_hour(now + timedelta(hours=2))
    assert any(a.startswith("tightened BTC/USDT") for a in actions)
    async with e.db.sessions() as s:
        pos = (await s.execute(select(PositionRow))).scalar_one()
    assert float(pos.stop) > stop0 and float(pos.stop) >= entry  # at least breakeven
    assert fake.positions[pos.broker_position_id].stop == pos.stop  # moved at the broker too
    more = pd.DataFrame({"open": entry, "high": entry, "low": entry, "close": entry * 1.06, "volume": 1.0},
                        index=[t0 + pd.Timedelta(hours=k) for k in range(3, 7)])
    fake.h1 = pd.concat([fake.h1, more])
    actions = await runner.on_hour(now + timedelta(hours=6))
    assert any("time exit" in a for a in actions)
    async with e.db.sessions() as s:
        pos = (await s.execute(select(PositionRow))).scalar_one()
    assert pos.status == "closed" and "time exit" in pos.exit_reason


async def test_runner_loads_full_history_once_then_only_recent_days(runner_env):
    e, fake, h1, now = runner_env
    await e.settings_store.update({"strategies_enabled": {"scripted": ["BTC/USDT"]}, "instruments": ["BTC/USDT"]}, "t")
    calls = []
    orig = fake.history

    async def spy(inst, tf, start, end):
        calls.append(end - start)
        return await orig(inst, tf, start, end)

    fake.history = spy
    runner = StrategyRunner(e, {"scripted": LastBarSignal(fire=False)})
    await runner.on_hour(now)
    await runner.on_hour(now + timedelta(hours=1))
    from tradeengine.runner import HISTORY_DAYS, REFRESH_DAYS

    assert calls == [timedelta(days=HISTORY_DAYS), timedelta(days=REFRESH_DAYS)]
