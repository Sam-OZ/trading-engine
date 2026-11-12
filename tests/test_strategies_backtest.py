"""Strategies (no look-ahead, regime alignment) and the backtester's fill model, metrics and walk-forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tradeengine.backtest import (
    CostModel,
    GateRules,
    Trade,
    judge,
    metrics,
    run_backtest,
    walk_forward,
)
from tradeengine.models import AssetKind, Side
from tradeengine.strategies import (
    STRATEGIES,
    WARMUP_BARS,
    Signal,
    build_frame,
    columns,
    h4_from_h1,
    trailed_stop,
)


def synthetic_h1(n: int = 4000, seed: int = 3, start: str = "2023-01-02") -> pd.DataFrame:
    """Alternating trend / range / squeeze-breakout regimes so all three strategies fire."""
    rng = np.random.default_rng(seed)
    drift = np.zeros(n)
    vol = np.full(n, 0.0010)
    block = 300
    for b in range(n // block + 1):
        sl = slice(b * block, (b + 1) * block)
        kind = b % 4
        if kind == 0:
            drift[sl] = 0.00025
        elif kind == 1:
            drift[sl] = 0.0
        elif kind == 2:
            drift[sl] = -0.00025
        else:
            vol[sl] = 0.0003
            drift[(b + 1) * block - 30 : (b + 1) * block] = 0.0015
    rets = drift + rng.normal(0, 1, n) * vol
    close = 1.1 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    spread = np.abs(rng.normal(0, 1, n)) * vol * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(100, 200, n) * (1 + 3 * (np.abs(rets) > 2 * vol))
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


H1 = synthetic_h1()


# ---------------------------------------------------------------- frame and strategies
def test_h4_context_uses_only_closed_four_hour_bars():
    df = build_frame(H1)
    h4 = h4_from_h1(H1)
    from tradeengine import indicators as ind

    ema50 = ind.ema(h4["close"], 50)
    # H1 bar opening 07:00 closes 08:00 -> the 04:00-08:00 H4 bar is complete and usable.
    t = df.index[2000]
    t = t.replace(hour=7)
    assert df.loc[t, "h4_ema50"] == pytest.approx(ema50.loc[t.replace(hour=4)])
    # H1 bar opening 06:00 closes 07:00 -> only the 00:00-04:00 H4 bar is complete.
    assert df.loc[t.replace(hour=6), "h4_ema50"] == pytest.approx(ema50.loc[t.replace(hour=0)])


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_signals_identical_with_or_without_future_data(name):
    strat = STRATEGIES[name]
    p = strat.default_params
    full = strat.prepare(H1, p)
    cols = columns(full)
    hits = [i for i in range(WARMUP_BARS, len(full)) if strat.signal_at(cols, i, p, True) is not None]
    assert hits, f"{name} never fired on synthetic data"
    probe = hits[:6] + list(range(WARMUP_BARS + 5, len(full), 700))
    for i in probe:
        part = strat.prepare(H1.iloc[: i + 1], p)
        a = strat.signal_at(cols, i, p, True)
        b = strat.signal_at(columns(part), len(part) - 1, p, True)
        assert a == b, (name, i)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_spot_crypto_gets_buy_signals_only(name):
    strat = STRATEGIES[name]
    cols = columns(strat.prepare(H1, strat.default_params))
    sigs = [strat.signal_at(cols, i, strat.default_params, False) for i in range(WARMUP_BARS, len(H1))]
    assert all(s.side is Side.BUY for s in sigs if s is not None)


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_signal_has_a_stop_on_the_right_side(name):
    strat = STRATEGIES[name]
    cols = columns(strat.prepare(H1, strat.default_params))
    for i in range(WARMUP_BARS, len(H1)):
        s = strat.signal_at(cols, i, strat.default_params, True)
        if s is None:
            continue
        if s.side is Side.BUY:
            assert s.stop < s.entry and (s.target is None or s.target > s.entry)
        else:
            assert s.stop > s.entry and (s.target is None or s.target < s.entry)


def test_trailed_stop_only_tightens():
    # long from 100, stop 98 (risk 2)
    assert trailed_stop(Side.BUY, 100, 98, 2, 101, 1, 2.0, 1.0) == 98  # +0.5R: nothing
    assert trailed_stop(Side.BUY, 100, 98, 2, 102, 1, 2.0, 1.0) == 100  # +1R: breakeven (trail 100 too)
    assert trailed_stop(Side.BUY, 100, 98, 2, 105, 1, 2.0, 1.0) == 103  # trail 105-2
    assert trailed_stop(Side.BUY, 100, 103, 2, 104, 1, 2.0, 1.0) == 103  # never loosens
    assert trailed_stop(Side.SELL, 100, 102, 2, 95, 1, 2.0, 1.0) == 97
    assert trailed_stop(Side.BUY, 100, 98, 2, 110, 1, None, None) == 98  # mean reversion: no trail


# ---------------------------------------------------------------- fill model
@dataclass
class Scripted:
    """Emits given signals at given bars; everything else is the real backtester."""

    signals: dict
    name: str = "scripted"

    def prepare(self, h1, params):
        return h1

    def signal_at(self, cols, i, p, allow_short):
        return self.signals.get(i)


def flat_frame(n: int, price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"open": price, "high": price + 0.5, "low": price - 0.5, "close": price, "volume": 1.0, "atr": 1.0}, index=idx)
    return df


NOCOST = CostModel()
I0 = WARMUP_BARS + 5


def test_entry_next_open_and_stop_fill_costs_one_r():
    df = flat_frame(I0 + 20)
    df.iloc[I0 + 3, df.columns.get_loc("low")] = 97.0
    strat = Scripted({I0: Signal(Side.BUY, 100, 98, None, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, NOCOST, True)
    assert pd.Timestamp(t.entry_time) == df.index[I0 + 1] and t.entry == 100
    assert t.exit == 98 and t.reason == "stop" and t.r == pytest.approx(-1.0)
    assert t.risk_amount == pytest.approx(50.0)  # 0.5% of 10,000


def test_spread_slippage_and_commission_are_charged():
    df = flat_frame(I0 + 20)
    df.iloc[I0 + 3, df.columns.get_loc("high")] = 104.5
    cost = CostModel(spread=0.2, slippage=0.1, commission_rate=0.001)
    strat = Scripted({I0: Signal(Side.BUY, 100, 98, 104, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, cost, True)
    assert t.entry == pytest.approx(100.2)  # open + half spread 0.1 + slippage 0.1
    assert t.exit == 104 and t.reason == "target"
    units = 50.0 / (100.2 - 98)
    assert t.units == pytest.approx(units)
    assert t.costs == pytest.approx(0.001 * units * 100.2 + 0.001 * units * 104)
    assert t.r < (104 - 100.2) / (100.2 - 98)  # commission reduces R


def test_same_bar_stop_and_target_assumes_stop():
    df = flat_frame(I0 + 20)
    df.iloc[I0 + 2, df.columns.get_loc("low")] = 97.0
    df.iloc[I0 + 2, df.columns.get_loc("high")] = 105.0
    strat = Scripted({I0: Signal(Side.BUY, 100, 98, 104, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, NOCOST, True)
    assert t.reason == "stop"


def test_gap_through_stop_fills_at_open():
    df = flat_frame(I0 + 20)
    df.iloc[I0 + 3, df.columns.get_loc("open")] = 96.0
    df.iloc[I0 + 3, df.columns.get_loc("low")] = 95.5
    strat = Scripted({I0: Signal(Side.SELL, 100, 102, None, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, NOCOST, True)
    assert t.reason in ("time", "end of data")  # short: a gap DOWN is in its favour
    strat = Scripted({I0: Signal(Side.BUY, 100, 98, None, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, NOCOST, True)
    assert t.reason == "stop (gap)" and t.exit == 96.0 and t.r == pytest.approx(-2.0)


def test_time_exit_and_funding_per_night():
    df = flat_frame(I0 + 80)
    cost = CostModel(funding_annual=0.0365)  # 0.01% of notional per night
    strat = Scripted({I0: Signal(Side.BUY, 100, 98, None, 1.0, "t", max_bars=48)})
    (t,) = run_backtest(df, "X", "USD", strat, {}, cost, True)
    assert t.reason == "time" and t.bars == 48
    nights = (pd.Timestamp(t.exit_time).normalize() - pd.Timestamp(t.entry_time).normalize()).days
    assert t.costs == pytest.approx(t.units * 100 * 0.0001 * nights)


def test_jpy_quote_converts_to_usd():
    df = flat_frame(I0 + 20, 150.0)
    df.iloc[I0 + 3, df.columns.get_loc("low")] = 148.0
    strat = Scripted({I0: Signal(Side.BUY, 150, 149, None, 1.0, "t", max_bars=50)})
    (t,) = run_backtest(df, "USD/JPY", "JPY", strat, {}, NOCOST, True)
    assert t.r == pytest.approx(-1.0, rel=0.01) and t.risk_amount == pytest.approx(50.0)


# ---------------------------------------------------------------- metrics, walk-forward, gate
def _trade(r: float, day: int) -> Trade:
    t0 = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=day)
    return Trade("X", "s", "p", "BUY", str(t0), str(t0 + pd.Timedelta(hours=5)), 1, 1, 1, None, 1, 50, r * 50, r, 0, 5, "x")


def test_metrics_profit_factor_and_drawdown():
    m = metrics([_trade(2, 0), _trade(-1, 1), _trade(-1, 2), _trade(3, 3)])
    assert m.trades == 4 and m.win_rate == 0.5 and m.profit_factor == pytest.approx(2.5)
    assert m.expectancy_r == pytest.approx(0.75)
    eq = np.cumprod([1.01, 0.995, 0.995, 1.015])
    assert m.max_drawdown_pct == pytest.approx((eq[0] - eq[2]) / eq[0] * 100)


def test_walk_forward_judges_only_unseen_data():
    strat = STRATEGIES["trend_pullback"]
    h1 = synthetic_h1(n=24 * 30 * 22, seed=5)  # ~22 months of H1
    res = walk_forward(h1, "EUR/USD", "USD", AssetKind.FX, strat, CostModel(spread=0.00010), train_months=6, test_months=2, min_is_trades=1)
    assert res.windows
    for w, nxt in zip(res.windows, res.windows[1:], strict=False):
        assert pd.Timestamp(nxt.train_start) > pd.Timestamp(w.train_start)
    tests = [(pd.Timestamp(w.train_end, tz="UTC"), pd.Timestamp(w.test_end, tz="UTC")) for w in res.windows]
    for t in res.oos_trades:
        e = pd.Timestamp(t.entry_time)
        assert any(a.normalize() <= e < b + pd.Timedelta(days=1) for a, b in tests)
    first_train_start = pd.Timestamp(res.windows[0].train_start, tz="UTC")
    assert all(pd.Timestamp(t.entry_time) >= first_train_start + pd.DateOffset(months=6) - pd.Timedelta(days=1) for t in res.oos_trades)


def test_gate_rules():
    from tradeengine.backtest import SymbolResult

    good = SymbolResult("A", "s", oos_trades=[_trade(1.5 if k % 2 else -1, k) for k in range(240)])
    weak = SymbolResult("B", "s", oos_trades=[_trade(-1, k) for k in range(40)])
    v = judge("s", [good, weak], GateRules())
    assert v.passed and v.enabled_symbols == ["A"]  # B excluded: PF < 1
    v = judge("s", [good], GateRules(min_trades=500))
    assert not v.passed and "needs 500" in v.reasons[0] and v.enabled_symbols == []
