"""Settings, IDs, sizing, kill switch."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from tradeengine.config import InfraConfig
from tradeengine.ids import client_order_id, leg_id
from tradeengine.models import MarketSpec, Side, Source
from tradeengine.settings_store import RuntimeSettings, SettingsStore
from tradeengine.sizing import conversion_rate, position_size, round_down, round_to_tick


# ---------------------------------------------------------------- config
def test_only_demo_mode_is_accepted(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "live")
    with pytest.raises(ValidationError):
        InfraConfig(_env_file=None)
    monkeypatch.setenv("TRADING_MODE", "demo")
    assert InfraConfig(_env_file=None).trading_mode == "demo"


def test_no_live_trading_urls_in_code():
    import tradeengine.brokers.binance as b
    import tradeengine.brokers.capital as c

    assert c.DEMO_REST.startswith("https://demo-api-capital")
    assert "api-capital.backend-capital.com" not in c.DEMO_REST.replace("demo-api-capital", "")
    assert b.DEMO_TRADE_REST == "https://demo-api.binance.com"
    src = open(c.__file__).read() + open(b.__file__).read()
    assert "https://api-capital.backend-capital.com" not in src


# ---------------------------------------------------------------- settings
async def test_settings_defaults_and_update(db):
    store = SettingsStore(db)
    s = await store.load()
    assert s.risk_per_trade_pct == D("0.5") and s.daily_loss_limit_pct == D("2.0") and s.max_drawdown_pct == D("10.0")
    assert s.claude_budget == D("2000") and s.claude_loss_floor == D("1700") and s.away_mode is False
    s2 = await store.update({"risk_per_trade_pct": "0.4", "away_mode": True}, "test")
    assert s2.risk_per_trade_pct == D("0.4")
    again = await store.load()
    assert again.risk_per_trade_pct == D("0.4") and again.away_mode is True


async def test_settings_reject_bad_values(db):
    store = SettingsStore(db)
    with pytest.raises(ValidationError):
        await store.update({"claude_loss_floor": "2500"}, "t")  # above budget
    with pytest.raises(ValidationError):
        await store.update({"risk_per_trade_pct": "5"}, "t")  # above 2% cap
    with pytest.raises(KeyError):
        await store.update({"no_such_setting": 1}, "t")
    assert (await store.load()).claude_loss_floor == D("1700")  # nothing saved


def test_away_schedule_time_format():
    with pytest.raises(ValidationError):
        RuntimeSettings(away_schedule_start="25:00")


# ---------------------------------------------------------------- ids
def test_client_order_id_deterministic_and_binance_safe():
    t = datetime(2026, 11, 12, 1, 0, tzinfo=UTC)
    a = client_order_id(Source.STRATEGY, "trend", "EUR/USD", Side.BUY, t)
    b = client_order_id(Source.STRATEGY, "trend", "EUR/USD", Side.BUY, t)
    c = client_order_id(Source.CLAUDE, "trend", "EUR/USD", Side.BUY, t)
    assert a == b != c
    pat = re.compile(r"^[\.A-Z\:/a-z0-9_-]{1,36}$")
    for x in (a, leg_id(a, "oco"), f"{a}-oco99"):
        assert pat.match(x)


# ---------------------------------------------------------------- sizing
def test_conversion_routes():
    rates = {"EUR/USD": D("1.10"), "AUD/USD": D("0.65"), "USD/JPY": D("150")}
    assert conversion_rate("USD", "USD", rates) == 1
    assert conversion_rate("USDT", "USD", rates) == 1
    assert conversion_rate("EUR", "USD", rates) == D("1.10")
    assert conversion_rate("USD", "AUD", rates) == 1 / D("0.65")
    assert conversion_rate("JPY", "AUD", rates) == (1 / D("150")) * (1 / D("0.65"))
    assert conversion_rate("CHF", "AUD", rates) is None


def test_position_size_fx_usd_account():
    spec = MarketSpec(D("100"), D("1"), D("0.00001"))
    r = position_size(D("1.1000"), D("1.0976"), D("50"), spec, D("1"))  # 24-pip stop, $50 risk
    assert r.size == D("20833")
    assert r.risk_at_stop <= D("50")


def test_position_size_jpy_quote_and_aud_account():
    spec = MarketSpec(D("100"), D("1"), D("0.001"))
    usd_jpy_to_aud = (1 / D("150")) * (1 / D("0.65"))
    r = position_size(D("150.00"), D("150.38"), D("50"), spec, usd_jpy_to_aud)
    assert r.size is not None and r.risk_at_stop <= D("50")
    assert r.risk_at_stop > D("49")


def test_min_size_and_min_notional_rejections():
    big_min = MarketSpec(D("100000"), D("1"), D("0.00001"))
    r = position_size(D("1.1000"), D("1.0900"), D("50"), big_min, D("1"))
    assert r.size is None and "minimum size" in r.reason
    spot = MarketSpec(D("0.00001"), D("0.00001"), D("0.01"), D("5"))
    r = position_size(D("60000"), D("59000"), D("0.05"), spot, D("1"))
    assert r.size is None and "minimum" in r.reason


def test_rounding():
    assert round_down(D("0.123456789"), D("0.00001")) == D("0.12345")
    assert round_to_tick(D("1.234567"), D("0.0001")) == D("1.2346")


# ---------------------------------------------------------------- kill switch
async def test_kill_switch_file_and_flag(state):
    assert (await state.is_halted())[0] is False
    state.kill_file.parent.mkdir(parents=True, exist_ok=True)
    state.kill_file.write_text("manual")
    assert (await state.is_halted()) == (True, "manual")
    state.kill_file.unlink()
    await state.engage_kill_switch("drawdown", "test")
    halted, why = await state.is_halted()
    assert halted and "drawdown" in why
    state.kill_file.unlink()  # DB flag alone still halts
    assert (await state.is_halted())[0] is True
    await state.release_kill_switch("test")
    assert (await state.is_halted())[0] is False


async def test_equity_markers(state):
    assert await state.start_of_day_equity("capital", "2026-11-12", D("10000")) == D("10000")
    assert await state.start_of_day_equity("capital", "2026-11-12", D("9500")) == D("10000")
    assert await state.update_peak_equity("capital", D("10100")) == D("10100")
    assert await state.update_peak_equity("capital", D("9900")) == D("10100")
