"""Round 2 live side: new instruments, gold in the risk manager, engine start-up with an unknown
instrument, the runner's history seeding and its act-only-in-the-first-hour rule, and the
monthly Dukascopy download."""

from __future__ import annotations

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
from test_risk import WEDNESDAY, ctx, pos
from test_round2 import trending_h1

from tradeengine import runner as runner_mod
from tradeengine.brokers.base import BrokerError
from tradeengine.cli import cross_check
from tradeengine.config import InfraConfig
from tradeengine.db import PositionMetaRow, PositionRow
from tradeengine.engine import Engine
from tradeengine.history import (
    bars_to_frame,
    check_price_range,
    download_dukascopy_h1,
    dukascopy_point,
    frame_to_bars,
    history_path,
    read_frame,
    write_bars,
)
from tradeengine.models import (
    DEFAULT_INSTRUMENTS,
    AssetKind,
    MarketSpec,
    Quote,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from tradeengine.risk import evaluate
from tradeengine.runner import StrategyRunner
from tradeengine.strategies import Signal
from tradeengine.trend import ChannelBreakoutD1

GOLD = DEFAULT_INSTRUMENTS["XAU/USD"]


# ------------------------------------------------------------------ instruments + risk
def test_round2_basket_is_in_the_instrument_list_and_allowed_by_default():
    from tradeengine.round2 import ROUND2_SYMBOLS
    from tradeengine.settings_store import RuntimeSettings

    assert set(ROUND2_SYMBOLS) <= set(DEFAULT_INSTRUMENTS)
    assert RuntimeSettings().instruments == ROUND2_SYMBOLS
    assert GOLD.kind is AssetKind.METAL and GOLD.broker_symbol == "GOLD" and GOLD.currencies == ("XAU", "USD")


def gold_ctx(**kw):
    c = ctx(**kw)
    c.tick_age_seconds["XAU/USD"] = 1.0
    c.specs["XAU/USD"] = MarketSpec(D("0.01"), D("0.01"), D("0.01"))
    return c


def buy_gold(**kw):
    base = dict(symbol="XAU/USD", side=Side.BUY, entry=D("2400"), stop=D("2370"), target=None, source=Source.STRATEGY,
                strategy="channel_breakout_d1", signal_time=WEDNESDAY)
    base.update(kw)
    return TradeIntent(**base)


def test_gold_counts_towards_the_usd_limit_and_keeps_forex_hours():
    assert evaluate(buy_gold(), GOLD, gold_ctx()).approved
    two_usd = [pos("GBP/USD"), pos("EUR/USD", ccy=("EUR", "USD"))]
    d = evaluate(buy_gold(), GOLD, gold_ctx(open_positions=two_usd))
    assert not d.approved and "USD" in d.reasons[0]
    saturday = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    d = evaluate(buy_gold(signal_time=saturday), GOLD, gold_ctx(now=saturday))
    assert not d.approved and "weekend" in d.reasons[0]
    # gold held, then a USD pair: the gold position counts against USD as well
    held = [pos("XAU/USD", kind=AssetKind.METAL, ccy=("XAU", "USD")), pos("GBP/USD")]
    from conftest import EURUSD
    from test_risk import buy_eur

    d = evaluate(buy_eur(), EURUSD, ctx(open_positions=held))
    assert not d.approved and "USD" in d.reasons[0]


# ------------------------------------------------------------------ engine start-up
class NoGold(FakeAdapter):
    def __init__(self, status=404):
        super().__init__(Venue.CAPITAL, "USD")
        self.status = status

    async def market_spec(self, inst):
        if inst.broker_symbol == "GOLD":
            raise BrokerError("Capital.com GET /api/v1/markets/GOLD failed (404): error.not-found.epic", status=self.status)
        return await super().market_spec(inst)


async def test_engine_leaves_out_an_instrument_the_broker_does_not_list(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path)
    e = Engine(cfg, adapters={Venue.CAPITAL: NoGold()})
    await e.start()
    try:
        assert "XAU/USD" in e.unavailable and "XAU/USD" not in e.specs
        active = {i.symbol for i in e.active_instruments()}
        assert "XAU/USD" not in active and {"EUR/USD", "GBP/JPY", "USD/CHF"} <= active
    finally:
        await e.stop()


async def test_engine_start_still_fails_loudly_on_other_broker_errors(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path)
    e = Engine(cfg, adapters={Venue.CAPITAL: NoGold(status=503)})
    with pytest.raises(BrokerError):
        await e.start()
    await e.stop()


# ------------------------------------------------------------------ runner
class HistoryFake(FakeAdapter):
    def __init__(self, venue: Venue, h1: pd.DataFrame, currency: str):
        super().__init__(venue, currency)
        self.h1 = h1
        self.calls: list[tuple] = []

    async def market_spec(self, inst):
        if inst.venue is Venue.BINANCE:
            return MarketSpec(D("0.00001"), D("0.00001"), D("0.01"), D("5"))
        return MarketSpec(D("1"), D("1"), D("0.00001"))

    async def quote(self, inst):
        last = float(self.h1["close"].iloc[-1])
        return Quote(inst.symbol, D(repr(last)) * D("0.9999"), D(repr(last)) * D("1.0001"), datetime.now(UTC))

    async def history(self, inst, tf, start, end):
        self.calls.append((inst.symbol, start, end))
        df = self.h1[(self.h1.index >= pd.Timestamp(start)) & (self.h1.index < pd.Timestamp(end))]
        return frame_to_bars(df, inst.symbol, tf)


async def _engine(tmp_path, adapters):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'r.db'}", data_dir=tmp_path)
    e = Engine(cfg, adapters=adapters)
    await e.start()
    return e


async def test_forex_history_starts_from_the_local_file_plus_recent_capital_bars(tmp_path):
    h1 = trending_h1("2023-01-01", "2024-06-01", seed=3)
    local = h1[h1.index < pd.Timestamp("2024-05-25", tz="UTC")]
    write_bars(history_path(tmp_path, "dukascopy", "EUR/USD", "H1"), frame_to_bars(local, "EUR/USD", "H1"))
    recent = h1.copy()
    recent.loc[recent.index >= pd.Timestamp("2024-05-20", tz="UTC"), "close"] *= 1.0001  # broker prices win on overlap
    fake = HistoryFake(Venue.CAPITAL, recent, "USD")
    e = await _engine(tmp_path, {Venue.CAPITAL: fake})
    try:
        r = StrategyRunner(e)
        now = datetime(2024, 5, 31, 12, 0, 30, tzinfo=UTC)
        df = await r._h1(DEFAULT_INSTRUMENTS["EUR/USD"], now)
        assert df.index[0] < pd.Timestamp("2023-01-03", tz="UTC")  # the local file's full span
        assert df.index[-1] == pd.Timestamp("2024-05-31 11:00", tz="UTC")
        assert df.loc[pd.Timestamp("2024-05-28 10:00", tz="UTC"), "close"] == pytest.approx(recent.loc["2024-05-28 10:00", "close"])
        assert [c[0] for c in fake.calls] == ["EUR/USD"] and fake.calls[0][2] - fake.calls[0][1] == timedelta(days=runner_mod.SEED_RECENT_DAYS)
    finally:
        await e.stop()


async def test_a_history_gap_is_filled_from_dukascopy_or_the_symbol_is_not_traded(tmp_path, monkeypatch):
    h1 = trending_h1("2023-01-01", "2024-06-01", seed=3)
    write_bars(history_path(tmp_path, "dukascopy", "EUR/USD", "H1"),
               frame_to_bars(h1[h1.index < pd.Timestamp("2024-04-01", tz="UTC")], "EUR/USD", "H1"))
    fake = HistoryFake(Venue.CAPITAL, h1, "USD")
    e = await _engine(tmp_path, {Venue.CAPITAL: fake})
    now = datetime(2024, 5, 31, 12, 0, 30, tzinfo=UTC)
    try:
        async def no_fill(client, symbol, start, end, **kw):
            return [], {}

        monkeypatch.setattr(runner_mod, "download_dukascopy_h1", no_fill)
        with pytest.raises(RuntimeError, match="history gap"):
            await StrategyRunner(e)._h1(DEFAULT_INSTRUMENTS["EUR/USD"], now)

        async def fill(client, symbol, start, end, **kw):
            part = h1[(h1.index >= pd.Timestamp(start, tz="UTC")) & (h1.index < pd.Timestamp(end, tz="UTC"))]
            return frame_to_bars(part, symbol, "H1"), {}

        monkeypatch.setattr(runner_mod, "download_dukascopy_h1", fill)
        df = await StrategyRunner(e)._h1(DEFAULT_INSTRUMENTS["EUR/USD"], now)
        assert df.index.to_series().diff().max() <= pd.Timedelta(days=3)
        on_disk = read_frame(history_path(tmp_path, "dukascopy", "EUR/USD", "H1"))
        assert on_disk.index[-1] >= pd.Timestamp("2024-05-15", tz="UTC")  # the fill was saved for next time
    finally:
        await e.stop()


@dataclass
class AlwaysBuyD1(ChannelBreakoutD1):
    name: str = "always_d1"
    min_history_days: int = 30
    exit_now: bool = False

    def signal_at(self, cols, i, p, allow_short):
        c, atr = float(cols["close"][i]), float(cols["atr"][i])
        return Signal(Side.BUY, c, c - 2 * atr, None, atr, "test", 3.0, None, None, 0.0)

    def exit_at(self, cols, i, side, p):
        return self.exit_now


@pytest.fixture
async def crypto_env(tmp_path):
    h1 = trending_h1("2024-01-01", "2024-04-01", seed=8, base=60000, crypto=True)
    fake = HistoryFake(Venue.BINANCE, h1, "USDT")
    e = await _engine(tmp_path, {Venue.BINANCE: fake})
    await e.settings_store.update({"strategies_enabled": {"always_d1": ["BTC/USDT"]}, "instruments": ["BTC/USDT"]}, "t")
    yield e, fake, h1
    await e.stop()


async def test_daily_strategy_acts_only_in_the_first_hour_after_its_bar_closes(crypto_env):
    e, fake, h1 = crypto_env
    strat = AlwaysBuyD1()
    # 02:00 UTC: two hourly bars have closed since the daily bar did -> its moment has passed
    late = datetime(2024, 3, 31, 2, 0, 30, tzinfo=UTC)
    assert await StrategyRunner(e, {"always_d1": strat}).on_hour(late) == []
    # 00:00 UTC: the daily bar has just closed -> act, and with the bar's own open time as the signal time
    fresh = datetime(2024, 3, 31, 0, 0, 30, tzinfo=UTC)
    actions = await StrategyRunner(e, {"always_d1": strat}).on_hour(fresh)
    assert actions == ["opened BUY BTC/USDT (always_d1)"]
    async with e.db.sessions() as s:
        meta = (await s.execute(select(PositionMetaRow))).scalar_one()
    assert pd.Timestamp(meta.signal_bar).tz_localize(None) == pd.Timestamp("2024-03-30 00:00")
    assert meta.max_bars == 0  # no time exit for round 2 strategies


async def test_signal_exit_closes_the_position_and_trail_uses_daily_atr(crypto_env):
    e, fake, h1 = crypto_env
    strat = AlwaysBuyD1()
    fresh = datetime(2024, 3, 30, 0, 0, 30, tzinfo=UTC)
    await StrategyRunner(e, {"always_d1": strat}).on_hour(fresh)
    strat.exit_now = True
    nxt = datetime(2024, 3, 31, 0, 0, 30, tzinfo=UTC)
    actions = await StrategyRunner(e, {"always_d1": strat}).on_hour(nxt)
    assert any("exit signal" in a for a in actions)
    async with e.db.sessions() as s:
        rows = (await s.execute(select(PositionRow).order_by(PositionRow.id))).scalars().all()
    assert rows[0].status == "closed" and "exit signal" in rows[0].exit_reason


# ------------------------------------------------------------------ Dukascopy hourly download
def _candles(base: datetime, hours: list[int], price: int) -> bytes:
    recs = b"".join(struct.pack(">IIIIIf", h * 3600, price, price + 10, price - 10, price + 20, 5.0) for h in hours)
    return lzma.compress(recs, format=lzma.FORMAT_ALONE)


@respx.mock
async def test_monthly_hourly_files_with_fallback_to_minutes_and_minutes_for_the_current_month():
    respx.get(url__regex=r".*/EURUSD/2024/00/BID_candles_hour_1\.bi5").mock(
        return_value=httpx.Response(200, content=_candles(datetime(2024, 1, 1), [0, 1, 24 * 30], 110000)))
    respx.get(url__regex=r".*/EURUSD/2024/01/BID_candles_hour_1\.bi5").mock(return_value=httpx.Response(404))
    minute = lzma.compress(struct.pack(">IIIIIf", 0, 110100, 110110, 110090, 110120, 1.0), format=lzma.FORMAT_ALONE)
    respx.get(url__regex=r".*/EURUSD/2024/0[12]/\d\d/BID_candles_min_1\.bi5").mock(return_value=httpx.Response(200, content=minute))
    async with httpx.AsyncClient() as client:
        bars, stats = await download_dukascopy_h1(client, "EUR/USD", date(2024, 1, 1), date(2024, 3, 3), today=date(2024, 3, 2))
    jan = [b for b in bars if b.open_time.month == 1]
    assert [b.open_time.hour for b in jan[:2]] == [0, 1] and jan[2].open_time.day == 31
    assert all(b.timeframe == "H1" for b in bars) and float(jan[0].close) == pytest.approx(1.1001)
    feb = [b for b in bars if b.open_time.month == 2]
    assert len(feb) == 29  # leap year: one bar per day built from minute files
    mar = [b for b in bars if b.open_time.month == 3]
    assert [b.open_time.day for b in mar] == [1, 2]  # the current month from minutes, nothing after "today"
    assert stats == {"monthly": 1, "from_minutes": 2}


def test_gold_scale_and_price_range_check():
    assert dukascopy_point("XAU/USD") == 1000 and dukascopy_point("GBP/JPY") == 1000 and dukascopy_point("USD/CHF") == 100000
    good = frame_to_bars(trending_h1("2024-01-01", "2024-01-05", base=2000.0), "XAU/USD", "H1")
    check_price_range("XAU/USD", good)
    wrong = frame_to_bars(trending_h1("2024-01-01", "2024-01-05", base=20.0), "XAU/USD", "H1")
    with pytest.raises(ValueError, match="price scale"):
        check_price_range("XAU/USD", wrong)


def test_cross_check_against_the_file_on_disk():
    a = trending_h1("2024-01-01", "2024-01-10")
    b = a.copy()
    n, mx, share = cross_check(a, b, 0.0001)
    assert n == len(a) and mx == 0 and share == 0
    b["close"] = b["close"] + 0.001
    n, mx, share = cross_check(a, b, 0.0001)
    assert share == 1.0 and mx == pytest.approx(0.001)
    assert cross_check(None, b, 0.0001) == (0, 0.0, 0.0)
    assert len(bars_to_frame(frame_to_bars(a, "EUR/USD", "H1"))) == len(a)


# ------------------------------------------------------------------ CLI: instruments + Dukascopy download
class SearchableNoGold(NoGold):
    async def search_markets(self, term):
        return [{"epic": "GOLD", "instrumentName": "Gold", "instrumentType": "COMMODITIES", "marketStatus": "TRADEABLE"}]


def test_instruments_command_reports_codes_spreads_and_candidates(tmp_path, monkeypatch, capsys):
    from tradeengine import cli

    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path)
    monkeypatch.setattr(cli, "_engine", lambda need_brokers=True: Engine(cfg, adapters={Venue.CAPITAL: SearchableNoGold()}))
    cli.main(["instruments"])
    out = capsys.readouterr().out
    assert "XAU/USD   capital code GOLD: NOT FOUND" in out and "candidate: GOLD" in out
    assert "EUR/USD   capital code EURUSD" in out and "spread now" in out
    assert "BTC/USDT  binance: no keys loaded" in out


def test_download_refuses_to_merge_hours_that_disagree_with_the_file_on_disk(tmp_path, monkeypatch, capsys):
    from tradeengine import cli

    h1 = trending_h1("2024-01-01", "2024-02-01")
    path = history_path(tmp_path, "dukascopy", "EUR/USD", "H1")
    write_bars(path, frame_to_bars(h1, "EUR/USD", "H1"))
    shifted = h1.copy()
    shifted["close"] += 0.002

    async def fake_download(client, symbol, start, end, **kw):
        return frame_to_bars(shifted, symbol, "H1"), {"monthly": 1, "from_minutes": 0}

    monkeypatch.setattr(cli, "download_dukascopy_h1", fake_download)
    cfg = InfraConfig(_env_file=None, data_dir=tmp_path)
    import asyncio

    asyncio.run(cli._download_dukascopy(cfg, "EUR/USD", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)))
    assert "NOT SAVED" in capsys.readouterr().out
    assert read_frame(path)["close"].iloc[5] == pytest.approx(h1["close"].iloc[5])  # file untouched

    async def same_download(client, symbol, start, end, **kw):
        return frame_to_bars(h1, symbol, "H1"), {"monthly": 1, "from_minutes": 0}

    monkeypatch.setattr(cli, "download_dukascopy_h1", same_download)
    asyncio.run(cli._download_dukascopy(cfg, "EUR/USD", datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 2, 1, tzinfo=UTC)))
    assert "matches the" in capsys.readouterr().out


async def test_no_action_on_stale_history(crypto_env):
    """If the broker returns no new bars, the last bar in memory is old: nothing is acted on."""
    e, fake, h1 = crypto_env
    strat = AlwaysBuyD1()
    fake.h1 = h1[h1.index < pd.Timestamp("2024-03-20", tz="UTC")]  # data stops 11 days before "now"
    now = datetime(2024, 3, 31, 0, 0, 30, tzinfo=UTC)
    assert await StrategyRunner(e, {"always_d1": strat}).on_hour(now) == []
    assert fake.open_calls == 0
