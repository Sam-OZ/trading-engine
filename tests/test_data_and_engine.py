"""Market data, history files, Dukascopy decoding, and the engine's trade path end to end."""

from __future__ import annotations

import lzma
import struct
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D

import pytest
from conftest import FakeAdapter
from sqlalchemy import select

from tradeengine.config import InfraConfig
from tradeengine.db import PositionRow
from tradeengine.engine import Engine, away_active
from tradeengine.history import (
    decode_dukascopy_candles,
    history_path,
    read_bars,
    write_bars,
)
from tradeengine.marketdata import (
    BarStore,
    MinuteBarBuilder,
    TickTracker,
    complete_only,
    resample,
)
from tradeengine.models import Bar, OrderStatus, Quote, Side, Source, TradeIntent, Venue
from tradeengine.settings_store import RuntimeSettings


def q(t: datetime, mid: str, symbol="EUR/USD") -> Quote:
    m = D(mid)
    return Quote(symbol, m - D("0.0001"), m + D("0.0001"), t)


# ---------------------------------------------------------------- market data
def test_minute_bars_from_quotes():
    b = MinuteBarBuilder()
    t0 = datetime(2026, 11, 11, 14, 0, 5, tzinfo=UTC)
    assert b.on_quote(q(t0, "1.1000")) is None
    assert b.on_quote(q(t0 + timedelta(seconds=20), "1.1010")) is None
    assert b.on_quote(q(t0 + timedelta(seconds=40), "1.0990")) is None
    bar = b.on_quote(q(t0 + timedelta(seconds=70), "1.1005"))
    assert bar.open_time == datetime(2026, 11, 11, 14, 0, tzinfo=UTC)
    assert (bar.open, bar.high, bar.low, bar.close) == (D("1.1000"), D("1.1010"), D("1.0990"), D("1.0990"))


def test_resample_aligns_to_utc_and_drops_incomplete():
    start = datetime(2026, 11, 11, 13, 58, tzinfo=UTC)
    bars = [Bar("X", "M1", start + timedelta(minutes=i), D(i), D(i) + 1, D(i) - 1, D(i) + D("0.5"), D(1)) for i in range(20)]
    m15 = resample(bars, "M15")
    assert [b.open_time.minute for b in m15] == [45, 0, 15]
    assert m15[1].open == D(2) and m15[1].close == D("16.5") and m15[1].volume == D(15)
    done = complete_only(m15, "M15", datetime(2026, 11, 11, 14, 18, tzinfo=UTC))
    assert [b.open_time.minute for b in done] == [45, 0]


def test_bar_store_dedupes_and_tick_tracker():
    s = BarStore(maxlen=3)
    t = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(5):
        s.add(Bar("X", "M1", t + timedelta(minutes=i), D(i), D(i), D(i), D(i)))
    s.add(Bar("X", "M1", t + timedelta(minutes=4), D(9), D(9), D(9), D(9)))  # correction of last bar
    s.add(Bar("X", "M1", t + timedelta(minutes=1), D(0), D(0), D(0), D(0)))  # stale, ignored
    assert [b.close for b in s.get("X", "M1")] == [D(2), D(3), D(9)]
    tt = TickTracker()
    tt.update(q(t, "1.1"))
    assert tt.mids() == {"EUR/USD": D("1.1")} and tt.age_seconds()["EUR/USD"] < 1


# ---------------------------------------------------------------- history files
def _bi5(records: list[tuple]) -> bytes:
    raw = b"".join(struct.pack(">IIIIIf", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_dukascopy_decode():
    data = _bi5([(0, 110000, 110020, 109990, 110030, 12.5), (60, 110020, 110020, 110020, 110020, 0.0), (120, 110020, 110010, 110005, 110025, 3.0)])
    bars = decode_dukascopy_candles(data, date(2026, 11, 11), "EUR/USD")
    assert len(bars) == 2  # flat zero-volume padding skipped
    b = bars[0]
    assert b.open_time == datetime(2026, 11, 11, tzinfo=UTC)
    assert (b.open, b.close, b.low, b.high) == (D("1.1"), D("1.1002"), D("1.0999"), D("1.1003"))
    jpy = decode_dukascopy_candles(_bi5([(0, 150000, 150100, 149900, 150200, 1.0)]), date(2026, 11, 11), "USD/JPY")
    assert jpy[0].open == D("150")
    assert decode_dukascopy_candles(b"", date(2026, 11, 14), "EUR/USD") == []


def test_dukascopy_sanity_check_catches_field_order_change():
    bad = _bi5([(0, 110000, 110020, 110030, 109990, 1.0)])  # low > high
    with pytest.raises(ValueError, match="sanity"):
        decode_dukascopy_candles(bad, date(2026, 11, 11), "EUR/USD")


def test_history_files_merge(tmp_path):
    t = datetime(2026, 11, 11, tzinfo=UTC)
    p = history_path(tmp_path, "binance", "BTC/USDT", "M1")
    write_bars(p, [Bar("BTC/USDT", "M1", t, D(1), D(2), D(0), D(1)), Bar("BTC/USDT", "M1", t + timedelta(minutes=1), D(1), D(2), D(0), D(1))])
    n = write_bars(p, [Bar("BTC/USDT", "M1", t + timedelta(minutes=1), D(5), D(6), D(4), D(5)), Bar("BTC/USDT", "M1", t + timedelta(minutes=2), D(1), D(2), D(0), D(1))])
    bars = read_bars(p, "BTC/USDT", "M1")
    assert n == 3 and bars[1].close == D(5) and p.name == "BTCUSDT_M1.csv.gz"


# ---------------------------------------------------------------- away mode
def test_away_schedule_overnight_perth():
    s = RuntimeSettings()  # 22:00-07:00 Australia/Perth (UTC+8)
    assert away_active(s, datetime(2026, 11, 12, 15, 0, tzinfo=UTC))  # 23:00 Perth
    assert away_active(s, datetime(2026, 11, 12, 22, 30, tzinfo=UTC))  # 06:30 Perth
    assert not away_active(s, datetime(2026, 11, 12, 2, 0, tzinfo=UTC))  # 10:00 Perth
    assert away_active(RuntimeSettings(away_mode=True), datetime(2026, 11, 12, 2, 0, tzinfo=UTC))
    assert not away_active(RuntimeSettings(away_schedule_enabled=False), datetime(2026, 11, 12, 15, 0, tzinfo=UTC))


# ---------------------------------------------------------------- engine end to end
@pytest.fixture
async def engine(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'e.db'}", data_dir=tmp_path)
    cap, bnb = FakeAdapter(Venue.CAPITAL), FakeAdapter(Venue.BINANCE, "USDT")
    e = Engine(cfg, adapters={Venue.CAPITAL: cap, Venue.BINANCE: bnb})
    await e.start()
    yield e
    await e.stop()


def intent(source=Source.STRATEGY, t=None) -> TradeIntent:
    t = t or datetime.now(UTC)
    return TradeIntent("EUR/USD", Side.BUY, D("1.1001"), D("1.0977"), D("1.1061"), source, "trend" if source is Source.STRATEGY else "claude", t)


def fx_open_now() -> bool:
    from tradeengine.risk import fx_market_state

    return fx_market_state(datetime.now(UTC), 2)[0]


@pytest.mark.skipif(not fx_open_now(), reason="FX weekend window: engine correctly refuses FX entries now")
async def test_engine_trade_path_end_to_end(engine):
    decision, outcome = await engine.trade(intent())
    assert decision.approved and outcome.status is OrderStatus.FILLED
    assert decision.size == D("20833")
    # A second signal on the same instrument (new signal time) is stopped by the per-instrument limit.
    again, dup = await engine.trade(intent())
    assert not again.approved and "already holding" in again.reasons[0] and dup is None
    async with engine.db.sessions() as s:
        assert len((await s.execute(select(PositionRow))).scalars().all()) == 1
    rep = await engine.reconciler.run_once()
    assert rep.matched == 1 and rep.clean


@pytest.mark.skipif(not fx_open_now(), reason="FX weekend window")
async def test_engine_claude_trade_waits_for_approval_unless_away(engine):
    await engine.settings_store.update({"away_schedule_enabled": False}, "t")
    decision, outcome = await engine.trade(intent(Source.CLAUDE))
    assert decision.approved and decision.needs_approval and outcome is None
    await engine.settings_store.update({"away_mode": True}, "t")
    decision, outcome = await engine.trade(intent(Source.CLAUDE))
    assert outcome is not None and outcome.status is OrderStatus.FILLED
    assert D("19") < decision.risk_amount <= D("20")


async def test_engine_drawdown_engages_kill_switch(engine):
    await engine.state.update_peak_equity("capital", D("12000"))  # equity 10000 = 16.7% down
    decision, outcome = await engine.trade(intent())
    assert not decision.approved and outcome is None
    assert (await engine.state.is_halted())[0]


async def test_engine_claude_floor_switches_trader_off(engine):
    async with engine.db.sessions() as s, s.begin():
        s.add(PositionRow(venue="capital", symbol="EUR/USD", side="BUY", size=D("10000"), entry_price=D("1.1"), broker_position_id="X",
                          source="claude", strategy="claude", status="closed", realised_pnl=D("-301"), closed_at=datetime.now(UTC) - timedelta(days=2),
                          opened_at=datetime.now(UTC) - timedelta(days=2)))
    decision, _ = await engine.trade(intent(Source.CLAUDE))
    assert not decision.approved and decision.disable_claude
    assert (await engine.settings_store.load()).claude_trader_enabled is False


async def test_emergency_flatten_closes_everything(engine):
    async with engine.db.sessions() as s, s.begin():
        s.add(PositionRow(venue="capital", symbol="EUR/USD", side="BUY", size=D("1000"), entry_price=D("1.1"), broker_position_id="D9",
                          source="strategy", strategy="x", status="open"))
    closed = await engine.emergency_flatten("test")
    assert closed == ["EUR/USD"] and (await engine.state.is_halted())[0]


async def test_binance_trades_the_virtual_balance_not_the_demo_default_balance(tmp_path):
    from tradeengine.models import AccountSnapshot

    class RichDemo(FakeAdapter):
        async def account(self):
            return AccountSnapshot(Venue.BINANCE, "USDT", D("250000"), D("400000"), D("250000"))

    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'v.db'}", data_dir=tmp_path)
    e = Engine(cfg, adapters={Venue.BINANCE: RichDemo(Venue.BINANCE, "USDT")})
    await e.start()
    try:
        acct = e.accounts[Venue.BINANCE]
        assert acct.equity == D("5000") and acct.available == D("5000")
        async with e.db.sessions() as s, s.begin():
            s.add(PositionRow(venue="binance", symbol="BTC/USDT", side="BUY", size=D("0.01"), entry_price=D("60000"), broker_position_id="1",
                              source="strategy", strategy="x", status="closed", realised_pnl=D("-25")))
            s.add(PositionRow(venue="binance", symbol="ETH/USDT", side="BUY", size=D("1"), entry_price=D("3000"), broker_position_id="2",
                              source="strategy", strategy="x", status="open"))
        await e.refresh_accounts()
        acct = e.accounts[Venue.BINANCE]
        assert acct.balance == D("4975") and acct.available == D("1975")
    finally:
        await e.stop()
