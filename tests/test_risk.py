"""Every risk rule, one test each."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal as D

import pytest
from conftest import BTCUSDT, EURUSD, USDJPY

from tradeengine.models import (
    AccountSnapshot,
    AssetKind,
    MarketSpec,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from tradeengine.risk import OpenPositionInfo, RiskContext, evaluate, fx_market_state
from tradeengine.settings_store import RuntimeSettings

WEDNESDAY = datetime(2026, 11, 11, 14, 0, tzinfo=UTC)  # mid-week, FX open


def ctx(**kw) -> RiskContext:
    base = dict(
        now=WEDNESDAY,
        settings=RuntimeSettings(),
        halted=False,
        halt_reason="",
        accounts={
            Venue.CAPITAL: AccountSnapshot(Venue.CAPITAL, "USD", D("10000"), D("10000"), D("9000")),
            Venue.BINANCE: AccountSnapshot(Venue.BINANCE, "USDT", D("10000"), D("10000"), D("10000")),
        },
        sod_equity={},
        peak_equity={},
        open_positions=[],
        tick_age_seconds={"EUR/USD": 1.0, "GBP/USD": 1.0, "USD/JPY": 1.0, "BTC/USDT": 1.0},
        rates={"EUR/USD": D("1.10"), "USD/JPY": D("150"), "GBP/USD": D("1.27")},
        specs={
            "EUR/USD": MarketSpec(D("100"), D("1"), D("0.00001")),
            "GBP/USD": MarketSpec(D("100"), D("1"), D("0.00001")),
            "USD/JPY": MarketSpec(D("100"), D("1"), D("0.001")),
            "BTC/USDT": MarketSpec(D("0.00001"), D("0.00001"), D("0.01"), D("5")),
        },
    )
    base.update(kw)
    return RiskContext(**base)


def buy_eur(**kw) -> TradeIntent:
    base = dict(
        symbol="EUR/USD", side=Side.BUY, entry=D("1.1000"), stop=D("1.0976"), target=D("1.1060"),
        source=Source.STRATEGY, strategy="trend", signal_time=WEDNESDAY,
    )
    base.update(kw)
    return TradeIntent(**base)


def pos(symbol="GBP/USD", venue=Venue.CAPITAL, kind=AssetKind.FX, ccy=("GBP", "USD"), source=Source.STRATEGY, notional=D("0")):
    return OpenPositionInfo(symbol, venue, kind, Side.BUY, ccy, source, notional)


def test_baseline_sizes_to_half_percent():
    d = evaluate(buy_eur(), EURUSD, ctx())
    assert d.approved, d.reasons
    assert d.size == D("20833")
    assert D("49") < d.risk_amount <= D("50")
    assert not d.needs_approval


@pytest.mark.parametrize(
    "intent,msg",
    [
        (buy_eur(stop=D("1.1010")), "below entry"),
        (buy_eur(side=Side.SELL, stop=D("1.0990"), target=D("1.09")), "above entry"),
        (buy_eur(target=D("1.0990")), "target must be above"),
        (buy_eur(stop=D("0")), "no stop"),
        (buy_eur(size_factor=D("1.5")), "size factor"),
    ],
)
def test_stop_and_target_sanity(intent, msg):
    d = evaluate(intent, EURUSD, ctx())
    assert not d.approved and msg in d.reasons[0]


def test_kill_switch_blocks():
    d = evaluate(buy_eur(), EURUSD, ctx(halted=True, halt_reason="manual"))
    assert not d.approved and "kill switch" in d.reasons[0]


def test_instrument_allowlist():
    s = RuntimeSettings(instruments=["BTC/USDT"])
    assert "allowed instrument" in evaluate(buy_eur(), EURUSD, ctx(settings=s)).reasons[0]


def test_spot_crypto_is_long_only():
    i = TradeIntent("BTC/USDT", Side.SELL, D("60000"), D("60600"), D("59000"), Source.STRATEGY, "x", WEDNESDAY)
    d = evaluate(i, BTCUSDT, ctx())
    assert not d.approved and "long-only" in d.reasons[0]


def test_stale_data():
    d = evaluate(buy_eur(), EURUSD, ctx(tick_age_seconds={"EUR/USD": 120.0}))
    assert not d.approved and "stale" in d.reasons[0]


def test_fx_weekend_rules():
    sat = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    fri_late = datetime(2026, 11, 13, 21, 30, tzinfo=UTC)  # 16:30 New York
    sun_open = datetime(2026, 11, 15, 23, 0, tzinfo=UTC)  # 18:00 New York Sunday
    assert fx_market_state(sat, 2)[0] is False
    assert fx_market_state(fri_late, 2)[0] is False
    assert fx_market_state(sun_open, 2)[0] is True
    assert fx_market_state(WEDNESDAY, 2)[0] is True
    assert "weekend" in evaluate(buy_eur(), EURUSD, ctx(now=sat)).reasons[0]


def test_crypto_trades_at_weekend():
    sat = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
    i = TradeIntent("BTC/USDT", Side.BUY, D("60000"), D("59400"), D("61200"), Source.STRATEGY, "x", sat)
    assert evaluate(i, BTCUSDT, ctx(now=sat)).approved


def test_daily_loss_limit():
    c = ctx(sod_equity={Venue.CAPITAL: D("10205")})  # down 2.01% today
    d = evaluate(buy_eur(), EURUSD, c)
    assert not d.approved and "daily loss" in d.reasons[0] and not d.halt


def test_drawdown_halts_engine():
    c = ctx(peak_equity={Venue.CAPITAL: D("11112")})  # 10.0% below peak
    d = evaluate(buy_eur(), EURUSD, c)
    assert not d.approved and d.halt and "drawdown" in d.reasons[0]


def test_max_open_positions_and_per_instrument():
    four = [pos(f"X{i}", ccy=("X", "Y")) for i in range(4)]
    assert "open positions" in evaluate(buy_eur(), EURUSD, ctx(open_positions=four)).reasons[0]
    same = [pos("EUR/USD", ccy=("EUR", "USD"))]
    assert "already holding" in evaluate(buy_eur(), EURUSD, ctx(open_positions=same)).reasons[0]


def test_currency_exposure_cap():
    two_usd = [pos("GBP/USD", ccy=("GBP", "USD")), pos("USD/JPY", ccy=("USD", "JPY"))]
    d = evaluate(buy_eur(), EURUSD, ctx(open_positions=two_usd))
    assert not d.approved and "in USD" in d.reasons[0]


def test_leverage_cap_shrinks_size():
    near_cap = [pos("GBP/USD", ccy=("GBP", "USD"), notional=D("40000"))]
    s = RuntimeSettings(max_positions_per_currency=3)
    d = evaluate(buy_eur(), EURUSD, ctx(open_positions=near_cap, settings=s))
    assert d.approved and d.size * D("1.1") <= D("10000")
    assert any("leverage" in r for r in d.reasons)
    full = [pos("GBP/USD", ccy=("GBP", "USD"), notional=D("50000"))]
    assert "leverage cap" in evaluate(buy_eur(), EURUSD, ctx(open_positions=full, settings=s)).reasons[0]


def test_jpy_pair_sizing_converts_to_account_currency():
    i = TradeIntent("USD/JPY", Side.SELL, D("150.000"), D("150.380"), D("149.240"), Source.STRATEGY, "x", WEDNESDAY)
    d = evaluate(i, USDJPY, ctx())
    assert d.approved and D("49") < d.risk_amount <= D("50")


def test_aud_account_needs_conversion_rate():
    aud = {Venue.CAPITAL: AccountSnapshot(Venue.CAPITAL, "AUD", D("15000"), D("15000"), D("15000"))}
    d = evaluate(buy_eur(), EURUSD, ctx(accounts=aud))
    assert not d.approved and "exchange rate" in d.reasons[0]
    d = evaluate(buy_eur(), EURUSD, ctx(accounts=aud, rates={"EUR/USD": D("1.1"), "AUD/USD": D("0.65")}))
    assert d.approved and D("74") < d.risk_amount <= D("75")  # 0.5% of A$15,000


def test_spot_limited_by_free_balance():
    poor = {Venue.BINANCE: AccountSnapshot(Venue.BINANCE, "USDT", D("10000"), D("10000"), D("100"))}
    i = TradeIntent("BTC/USDT", Side.BUY, D("60000"), D("59400"), None, Source.STRATEGY, "x", WEDNESDAY)
    d = evaluate(i, BTCUSDT, ctx(accounts=poor))
    assert d.approved and d.size * D("60000") <= D("100")


# ---------------------------------------------------------------- Claude sleeve
def claude(**kw) -> TradeIntent:
    return buy_eur(source=Source.CLAUDE, strategy="claude", **kw)


def test_claude_risk_capped_at_20_and_needs_approval_when_present():
    d = evaluate(claude(), EURUSD, ctx())
    assert d.approved and D("19") < d.risk_amount <= D("20")
    assert d.needs_approval
    d2 = evaluate(claude(), EURUSD, ctx(away_active=True))
    assert d2.approved and not d2.needs_approval


def test_claude_switched_off():
    s = RuntimeSettings(claude_trader_enabled=False)
    assert "switched off" in evaluate(claude(), EURUSD, ctx(settings=s)).reasons[0]


def test_claude_floor_disables_trader():
    d = evaluate(claude(), EURUSD, ctx(claude_realised_total=D("-300")))
    assert not d.approved and d.disable_claude and "floor" in d.reasons[0]


def test_claude_daily_limit_and_remaining_allowance():
    assert "daily loss" in evaluate(claude(), EURUSD, ctx(claude_pnl_today=D("-60"))).reasons[0]
    d = evaluate(claude(), EURUSD, ctx(claude_pnl_today=D("-50")))  # only $10 left today
    assert d.approved and d.risk_amount <= D("10")


def test_claude_floor_headroom_limits_risk():
    d = evaluate(claude(), EURUSD, ctx(claude_realised_total=D("-290")))  # $10 above floor
    assert d.approved and d.risk_amount <= D("10")


def test_claude_position_and_trade_counts():
    two = [pos("GBP/USD", ccy=("GBP", "CHF"), source=Source.CLAUDE), pos("BTC/USDT", Venue.BINANCE, AssetKind.CRYPTO_SPOT, ("BTC", "USDT"), Source.CLAUDE)]
    assert "Claude already has 2" in evaluate(claude(), EURUSD, ctx(open_positions=two)).reasons[0]
    assert "6 trades" in evaluate(claude(), EURUSD, ctx(claude_trades_today=6)).reasons[0]


def test_claude_still_bound_by_engine_limits():
    c = ctx(halted=True, halt_reason="x", away_active=True)
    assert not evaluate(claude(), EURUSD, c).approved
    i = replace(claude(), stop=D("0"))
    assert "no stop" in evaluate(i, EURUSD, ctx(away_active=True)).reasons[0]


