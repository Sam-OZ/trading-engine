"""Round 2: daily/4-hour bars, the three trend strategies, the portfolio simulator, the walk-forward
choice with sit-out, the development/holdout drivers and the one-look lock."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from tradeengine import round2 as r2
from tradeengine.backtest import CostModel, Trade, run_backtest
from tradeengine.history import frame_to_bars, history_path, write_bars
from tradeengine.models import DEFAULT_INSTRUMENTS, AssetKind, Side, Venue
from tradeengine.portfolio import Leg, Limits, RateBook, portfolio_metrics, simulate
from tradeengine.settings_store import RuntimeSettings
from tradeengine.strategies import STRATEGIES, Signal, columns, trailed_stop
from tradeengine.trend import (
    ROUND2_STRATEGIES,
    BreakoutH4DailyFilter,
    ChannelBreakoutD1,
    TimeSeriesMomentumD1,
    daily_from_h1,
    h4_bars,
    prep_key,
    prepare,
)

NY = ZoneInfo("America/New_York")


def fx_open(idx: pd.DatetimeIndex) -> np.ndarray:
    """Forex trades Sunday 17:00 to Friday 17:00 New York time."""
    ny = idx.tz_convert(NY)
    wd, hr = ny.day_of_week, ny.hour
    return ~((wd == 5) | ((wd == 6) & (hr < 17)) | ((wd == 4) & (hr >= 17)))


def trending_h1(start: str, end: str, seed: int = 1, base: float = 1.1, crypto: bool = False, vol: float = 0.0015,
                drift: float = 0.0002, regime_hours: int = 24 * 50) -> pd.DataFrame:
    """Hourly bars with alternating up / flat / down regimes, so trend rules both win and lose."""
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    if not crypto:
        idx = idx[fx_open(idx)]
    n = len(idx)
    rng = np.random.default_rng(seed)
    regime = (np.arange(n) // regime_hours) % 3
    mu = np.where(regime == 0, drift, np.where(regime == 1, 0.0, -drift))
    rets = mu + rng.normal(0, vol, n)
    close = base * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base], close[:-1]])
    wick = np.abs(rng.normal(0, vol / 2, n)) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick,
                         "close": close, "volume": rng.uniform(100, 200, n)}, index=pd.DatetimeIndex(idx, name="open_time"))


# ------------------------------------------------------------------ bars
def test_fx_daily_sessions_close_at_17_new_york_through_dst():
    h1 = trending_h1("2024-03-01", "2024-03-20")  # US clocks change on 10 Mar 2024
    d = daily_from_h1(h1, crypto=False)
    assert (d["weekday"] < 5).all()  # no weekend sessions
    closes_ny = pd.DatetimeIndex(d["closes_at"]).tz_convert(NY)
    assert (closes_ny.hour == 17).all() and (closes_ny.minute == 0).all()
    opens_ny = d.index.tz_convert(NY)
    assert (opens_ny.hour == 17).all()
    # Monday's session opens on Sunday 17:00 New York (21:00 UTC in summer, 22:00 in winter)
    monday = d[d["weekday"] == 0]
    assert (monday.index.tz_convert(NY).day_of_week == 6).all()
    utc_hours = set(pd.DatetimeIndex(d["closes_at"]).hour)
    assert utc_hours == {21, 22}
    # OHLC aggregate the session's hours
    first = d.iloc[1]
    hours = h1[(h1.index >= d.index[1]) & (h1.index < first["closes_at"])]
    assert first["open"] == hours["open"].iloc[0] and first["close"] == hours["close"].iloc[-1]
    assert first["high"] == hours["high"].max() and first["low"] == hours["low"].min()


def test_only_completed_daily_and_h4_bars_are_kept():
    h1 = trending_h1("2024-06-03", "2024-06-08")
    cut = h1[h1.index < pd.Timestamp("2024-06-05 15:00", tz="UTC")]  # Wednesday mid-session
    d = daily_from_h1(cut, crypto=False)
    assert pd.Timestamp(d["closes_at"].iloc[-1]) <= cut.index[-1] + pd.Timedelta(hours=1)
    f = h4_bars(cut)
    assert (f["closes_at"] <= cut.index[-1] + pd.Timedelta(hours=1)).all()
    assert f.index[-1] == pd.Timestamp("2024-06-05 08:00", tz="UTC")  # 12:00 bar is still open


def test_crypto_daily_bars_are_utc_days_seven_days_a_week():
    h1 = trending_h1("2024-01-01", "2024-01-15", crypto=True, base=40000)
    d = daily_from_h1(h1, crypto=True)
    assert len(d) == 14 and set(d["weekday"]) == set(range(7))
    assert (d.index.hour == 0).all() and (pd.DatetimeIndex(d["closes_at"]) - d.index == pd.Timedelta(days=1)).all()


# ------------------------------------------------------------------ strategies
@pytest.fixture(scope="module")
def fx_long():
    return trending_h1("2019-01-01", "2022-01-01", seed=7)


@pytest.mark.parametrize("name", list(ROUND2_STRATEGIES))
def test_round2_signals_identical_with_or_without_future_data(fx_long, name):
    s = ROUND2_STRATEGIES[name]
    p = s.default_params
    full = prepare(s, fx_long, p, False)
    cols = columns(full)
    ready = np.flatnonzero(full["ready"].to_numpy(dtype=bool))
    fired = 0
    for i in ready[:: max(1, len(ready) // 25)][:25]:
        cut = fx_long[fx_long.index < pd.Timestamp(full["closes_at"].iloc[i])]
        part = prepare(s, cut, p, False)
        assert part.index[-1] == full.index[i]  # the bar is complete with data up to its close
        a = s.signal_at(cols, i, p, True)
        b = s.signal_at(columns(part), len(part) - 1, p, True)
        assert (a is None) == (b is None)
        if a is not None:
            fired += 1
            assert a.side == b.side and a.stop == pytest.approx(b.stop) and a.entry == pytest.approx(b.entry)
    signals = sum(s.signal_at(cols, int(i), p, True) is not None for i in ready)
    assert signals > 5  # the rules do fire on trending data


@pytest.mark.parametrize("name", list(ROUND2_STRATEGIES))
def test_round2_spot_long_only_and_stops_on_the_right_side(fx_long, name):
    s = ROUND2_STRATEGIES[name]
    f = prepare(s, fx_long, s.default_params, False)
    cols = columns(f)
    for i in range(len(f)):
        sig = s.signal_at(cols, i, s.default_params, False)
        if sig is not None:
            assert sig.side is Side.BUY and sig.stop < sig.entry and sig.target is None and sig.max_bars is None
            assert sig.trail_activate_r == 0.0


def test_tsmom_checks_only_on_fridays_and_exits_on_a_sign_flip(fx_long):
    s = TimeSeriesMomentumD1()
    p = {"lookback": 63, "stop_atr": 3.0, "trail_atr": 4.0}
    f = s.prepare(fx_long, p)
    cols = columns(f)
    days = [int(f["weekday"].iloc[i]) for i in range(len(f)) if s.signal_at(cols, i, p, True) is not None]
    assert days and set(days) == {4}
    flips = [i for i in range(len(f)) if s.exit_at(cols, i, Side.BUY, p)]
    assert flips and all(f["ret"].iloc[i] < 0 and f["weekday"].iloc[i] == 4 for i in flips)


def test_h4_daily_filter_uses_only_closed_daily_bars(fx_long):
    s = BreakoutH4DailyFilter()
    f = s.prepare(fx_long, s.default_params)
    d = daily_from_h1(fx_long, crypto=False)
    from tradeengine import indicators as ind

    ema = ind.ema(d["close"], 200)
    for i in range(len(f) - 1, len(f) - 400, -37):
        closed = d["closes_at"] <= f["closes_at"].iloc[i]
        j = np.flatnonzero(closed.to_numpy())[-1]
        assert f["d_close"].iloc[i] == d["close"].iloc[j]
        assert f["d_ema200"].iloc[i] == pytest.approx(ema.iloc[j])


def test_trail_can_start_at_once():
    # activate_r = 0: the ATR trail tightens as soon as the best close allows it
    assert trailed_stop(Side.BUY, 100, 94, 6, 101, 1.0, 3.0, None, 0.0) == 98
    assert trailed_stop(Side.BUY, 100, 94, 6, 101, 1.0, 3.0, None, 1.0) == 94  # round 1 behaviour unchanged
    assert trailed_stop(Side.SELL, 100, 106, 6, 99, 1.0, 3.0, None, 0.0) == 102


# ------------------------------------------------------------------ portfolio simulator
EUR = DEFAULT_INSTRUMENTS["EUR/USD"]
COST = CostModel(spread=0.0001, slippage=0.00005, funding_annual=0.06)


def leg_for(sym: str, h1: pd.DataFrame, strategy, params: dict, start, end, kind=AssetKind.FX, venue=Venue.CAPITAL,
            base="EUR", quote="USD", cost=COST) -> Leg:
    frames = {prep_key(strategy, params): prepare(strategy, h1, params, kind is AssetKind.CRYPTO_SPOT)}
    return Leg(sym, kind, venue, base, quote, cost, frames, [(start, end, params)], prep_of=lambda q: prep_key(strategy, q))


def test_single_leg_without_limits_reproduces_the_round1_backtester():
    from test_strategies_backtest import synthetic_h1

    h1 = synthetic_h1(n=4000, seed=3)
    strat = STRATEGIES["breakout"]
    p = strat.default_params
    frame = strat.prepare(h1, p)
    ref = run_backtest(frame, "EUR/USD", "USD", strat, p, COST, True)
    from tradeengine.strategies import WARMUP_BARS

    leg = Leg("EUR/USD", AssetKind.FX, Venue.CAPITAL, "EUR", "USD", COST, {"k": frame},
              [(frame.index[0], frame.index[-1] + pd.Timedelta(hours=1), p)], prep_of=lambda q: "k")
    res = simulate(strat, [leg], RateBook({}), frame.index[WARMUP_BARS], frame.index[-1] + pd.Timedelta(hours=1), None)
    assert len(ref) > 5 and len(res.trades) == len(ref)
    for a, b in zip(ref, res.trades, strict=True):
        assert (a.entry_time, a.exit_time, a.side, a.reason) == (b.entry_time, b.exit_time, b.side, b.reason)
        assert b.r == pytest.approx(a.r, rel=1e-9) and b.pnl == pytest.approx(a.pnl, rel=1e-9)


@dataclass
class EveryBar:
    """Signals a buy on every bar: stresses the limits."""

    name: str = "every_bar"
    timeframe: str = "D1"
    prep_keys: tuple = ()
    default_params: dict = field(default_factory=dict)
    stop_pct: float = 0.01
    bars: str = "D1"

    def prepare(self, h1, params, crypto=False):
        d = daily_from_h1(h1, crypto) if self.bars == "D1" else h4_bars(h1)
        from tradeengine import indicators as ind

        d["atr"] = ind.atr(d["high"], d["low"], d["close"], 5)
        d["ready"] = d["atr"].notna()
        return d

    def signal_at(self, cols, i, p, allow_short):
        c = cols["close"][i]
        return Signal(Side.BUY, c, c * (1 - self.stop_pct), None, cols["atr"][i], "test", None, None, 3, 0.0)

    def exit_at(self, cols, i, side, p):
        return False


def _concurrency(trades: list[Trade], symbols_of=None) -> int:
    ev = sorted([(pd.Timestamp(t.entry_time), 1) for t in trades] + [(pd.Timestamp(t.exit_time), -1) for t in trades],
                key=lambda x: (x[0], x[1]))
    n = peak = 0
    for _, d in ev:
        n += d
        peak = max(peak, n)
    return peak


def test_portfolio_respects_max_open_and_per_currency_limits():
    strat = EveryBar()
    start, end = pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp("2020-12-01", tz="UTC")
    legs, series = [], {}
    pairs = [("EUR/USD", "EUR", "USD", 1.1), ("GBP/USD", "GBP", "USD", 1.3), ("AUD/USD", "AUD", "USD", 0.7),
             ("USD/CHF", "USD", "CHF", 0.95), ("EUR/GBP", "EUR", "GBP", 0.85), ("NZD/USD", "NZD", "USD", 0.6)]
    for k, (sym, b, q, px) in enumerate(pairs):
        h1 = trending_h1("2020-01-01", "2021-01-01", seed=k, base=px)
        series[sym] = h1["close"]
        legs.append(leg_for(sym, h1, strat, {}, start, end, base=b, quote=q))
    res = simulate(strat, legs, RateBook(series), start, end, Limits(max_open=4, per_currency=2, daily_loss_pct=50, max_dd_pct=90))
    assert res.trades
    # never more than 4 at once, never more than 2 holding USD (or any currency)
    assert _concurrency(res.trades) <= 4
    cur = {s: (b, q) for s, b, q, _ in pairs}
    for ccy in ("USD", "EUR", "GBP"):
        assert _concurrency([t for t in res.trades if ccy in cur[t.symbol]]) <= 2
    assert res.rejections["max positions per currency (USD)"] > 0


def test_kill_switch_stops_new_entries_after_the_drawdown_limit():
    strat = EveryBar(stop_pct=0.002)  # tight stops in a falling market: a fast losing streak
    h1 = trending_h1("2020-01-01", "2021-01-01", seed=3, drift=-0.0006, regime_hours=10**9)
    start, end = pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")
    leg = leg_for("EUR/USD", h1, strat, {}, start, end)
    res = simulate(strat, [leg], RateBook({}), start, end, Limits(risk_pct=0.01, max_dd_pct=5, daily_loss_pct=50))
    assert res.halted_at is not None and res.rejections["kill switch (drawdown halt)"] > 0
    halt_time = pd.Timestamp(res.halted_at.split(" at ")[1])
    assert all(pd.Timestamp(t.entry_time) <= halt_time for t in res.trades)
    m = portfolio_metrics(res)
    assert m.max_drawdown_pct >= 5.0


def test_daily_loss_limit_blocks_the_rest_of_the_day():
    strat = EveryBar(stop_pct=0.0005, bars="H4")  # several entries a day, so a bad morning can block the afternoon
    start, end = pd.Timestamp("2020-02-01", tz="UTC"), pd.Timestamp("2020-06-01", tz="UTC")
    legs = [leg_for(s, trending_h1("2020-01-01", "2020-07-01", seed=k, drift=-0.0004, regime_hours=10**9), strat, {}, start, end,
                    base=s[:3], quote=s[4:]) for k, s in enumerate(["EUR/USD", "GBP/JPY"])]
    res = simulate(strat, legs, RateBook({}), start, end, Limits(risk_pct=0.01, daily_loss_pct=0.5, max_dd_pct=99))
    assert res.rejections["daily loss limit"] > 0


def test_mark_to_market_drawdown_is_at_least_the_closed_trade_drawdown():
    strat = ChannelBreakoutD1()
    p = strat.default_params
    h1 = trending_h1("2018-01-01", "2022-01-01", seed=11)
    start, end = pd.Timestamp("2018-09-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC")
    res = simulate(strat, [leg_for("EUR/USD", h1, strat, p, start, end)], RateBook({}), start, end, Limits(max_dd_pct=99, daily_loss_pct=99))
    m = portfolio_metrics(res)
    closed = 10_000 + np.cumsum([t.pnl for t in sorted(res.trades, key=lambda t: t.exit_time)])
    peak = np.maximum.accumulate(np.concatenate([[10_000], closed]))[1:]
    closed_dd = float(((peak - closed) / peak).max() * 100)
    assert res.trades and m.max_drawdown_pct >= closed_dd - 1e-9


def test_signal_exit_fills_at_the_next_open_and_can_reverse():
    strat = TimeSeriesMomentumD1()
    p = {"lookback": 63, "stop_atr": 3.0, "trail_atr": 4.0}
    h1 = trending_h1("2018-01-01", "2021-01-01", seed=7)
    start, end = pd.Timestamp("2018-08-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC")
    res = simulate(strat, [leg_for("EUR/USD", h1, strat, p, start, end)], RateBook({}), start, end, None)
    exits = [t for t in res.trades if t.reason == "signal exit"]
    assert exits
    f = strat.prepare(h1, p)
    for t in exits:
        i = f.index.get_loc(pd.Timestamp(t.exit_time))
        assert f["weekday"].iloc[i - 1] == 4  # decided at Friday's close, filled at the next open
    reversals = [(a, b) for a, b in zip(res.trades, res.trades[1:], strict=False) if a.reason == "signal exit" and b.entry_time == a.exit_time]
    assert reversals and all(a.side != b.side for a, b in reversals)


def test_rate_book_converts_without_look_ahead():
    idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    rb = RateBook({"USD/JPY": pd.Series([150.0, 151.0, 152.0], index=idx)})
    assert rb.to_usd("JPY", idx[1]) == pytest.approx(1 / 150.0)  # the 01:00 bar has not closed at 01:00
    assert rb.to_usd("JPY", idx[2]) == pytest.approx(1 / 151.0)
    assert rb.to_usd("USDT", idx[2]) == 1.0


# ------------------------------------------------------------------ walk-forward choice
def _t(entry: str, exit_: str, r: float) -> Trade:
    return Trade("X", "s", "p", "BUY", entry, exit_, 1, 1, 1, None, 1, 1, r, r, 0, 1, "x")


def test_choose_windows_counts_only_trades_closed_inside_training_and_sits_out_losers():
    first = pd.Timestamp("2020-01-01", tz="UTC")
    good = [_t(f"2020-{m:02d}-01 00:00:00+00:00", f"2020-{m:02d}-05 00:00:00+00:00", 2.0 if m % 2 else -1.0) for m in range(1, 13)]
    leaky = [_t("2020-11-20 00:00:00+00:00", "2021-02-01 00:00:00+00:00", 50.0)] + [
        _t(f"2020-{m:02d}-01 00:00:00+00:00", f"2020-{m:02d}-05 00:00:00+00:00", -1.0) for m in range(1, 7)]
    ws = r2.choose_windows({"good": good, "leaky": leaky}, first, pd.Timestamp("2021-07-01", tz="UTC"), 12, 6, 6, 1.0)
    assert ws[0].params_id == "good"  # the leaky set's +50R exits after training ends: it does not count
    losers = {"bad": [_t(f"2020-{m:02d}-01 00:00:00+00:00", f"2020-{m:02d}-05 00:00:00+00:00", -1.0 if m % 3 else 1.0) for m in range(1, 13)]}
    ws = r2.choose_windows(losers, first, pd.Timestamp("2021-07-01", tz="UTC"), 12, 6, 6, 1.0)
    assert ws[0].params_id is None and "sit out" in ws[0].note
    ws = r2.choose_windows({"few": good[:3]}, first, pd.Timestamp("2021-07-01", tz="UTC"), 12, 6, 6, 1.0)
    assert ws[0].params_id is None and "completed training trades" in ws[0].note


def test_schedule_is_clipped_to_the_period():
    ws = [r2.Window("2020-01-01", "2021-01-01", "2021-07-01", "a", 10, 1.5), r2.Window("2020-07-01", "2021-07-01", "2022-01-01", None, 9, 0.8)]
    sched = r2.schedule_for(ws, pd.Timestamp("2021-03-01"), pd.Timestamp("2021-10-01"), {"a": {"x": 1}})
    assert sched == [(pd.Timestamp("2021-03-01"), pd.Timestamp("2021-07-01"), {"x": 1}),
                     (pd.Timestamp("2021-07-01"), pd.Timestamp("2021-10-01"), None)]


# ------------------------------------------------------------------ drivers, CLI and the one-look lock
@pytest.fixture
def small_round(monkeypatch, tmp_path):
    """Round 2 on three instruments and a shortened calendar, so the whole pipeline runs in seconds."""
    monkeypatch.setattr(r2, "ROUND2_SYMBOLS", ["EUR/USD", "USD/JPY", "BTC/USDT"])
    monkeypatch.setattr(r2, "DATA_FROM_FX", pd.Timestamp("2016-01-01", tz="UTC"))
    monkeypatch.setattr(r2, "DATA_FROM_CRYPTO", pd.Timestamp("2016-01-01", tz="UTC"))
    monkeypatch.setattr(r2, "DEV_END", pd.Timestamp("2020-01-01", tz="UTC"))
    monkeypatch.setattr(r2, "HOLDOUT_END", pd.Timestamp("2021-01-01", tz="UTC"))
    for strat in ROUND2_STRATEGIES.values():  # 12-month training windows on synthetic data hold fewer trades
        monkeypatch.setattr(strat, "min_is_trades", 2)
    data = tmp_path / "data"
    for k, (sym, src, px, crypto) in enumerate([("EUR/USD", "dukascopy", 1.1, False), ("USD/JPY", "dukascopy", 110.0, False),
                                                 ("BTC/USDT", "binance", 9000.0, True)]):
        h1 = trending_h1("2016-01-01", "2021-03-01", seed=20 + k, base=px, crypto=crypto)
        write_bars(history_path(data, src, sym, "H1"), frame_to_bars(h1, sym, "H1"))
    return data


def _settings():
    return RuntimeSettings(round2_walk_forward={"channel_breakout_d1": [12, 6], "tsmom_d1": [12, 6], "breakout_h4_dfilter": [12, 6]})


def test_dev_run_never_sees_holdout_data_and_freezes_a_basket(small_round):
    runs, verdicts, notes = r2.run_dev(small_round, _settings(), DEFAULT_INSTRUMENTS, progress=lambda *_: None)
    assert [r.strategy for r in runs] == list(ROUND2_STRATEGIES)
    for run in runs:
        assert run.trades and all(pd.Timestamp(t.exit_time) < r2.DEV_END for t in run.trades)
        assert run.configurations == len(r2.param_grid(ROUND2_STRATEGIES[run.strategy].grid)) * 3
        v = verdicts[run.strategy]
        for sym in v["basket"]:
            n, p, _ = run.per_symbol[sym]
            assert n >= r2.RULES.basket_min_trades and p >= r2.RULES.basket_min_pf
        btc = [t for t in run.trades if t.symbol == "BTC/USDT"]
        assert all(t.side == "BUY" for t in btc)
    md = r2.render("dev", runs, verdicts, notes, "fp", [])
    assert "## Verdict" in md and "Frozen basket" in md and "configurations tested" in md


def test_holdout_uses_only_the_frozen_basket_and_only_holdout_months(small_round):
    dev = {"kind": "round2-dev", "strategies": {"channel_breakout_d1": {"passed": True, "basket": ["EUR/USD"]},
                                                "tsmom_d1": {"passed": False, "basket": []}}}
    runs, verdicts, _ = r2.run_holdout(small_round, _settings(), DEFAULT_INSTRUMENTS, dev, progress=lambda *_: None)
    assert [r.strategy for r in runs] == ["channel_breakout_d1"]  # failed development = never looked at
    run = runs[0]
    assert {t.symbol for t in run.trades} <= {"EUR/USD"}
    assert all(r2.DEV_END <= pd.Timestamp(t.entry_time) < r2.HOLDOUT_END for t in run.trades)
    v = verdicts["channel_breakout_d1"]
    assert v["basket"] == ["EUR/USD"] and (v["enabled_symbols"] == [] or v["passed"])


def test_fingerprint_changes_with_settings_that_change_results():
    a = r2.fingerprint(RuntimeSettings())
    assert a == r2.fingerprint(RuntimeSettings())
    assert a != r2.fingerprint(RuntimeSettings(risk_per_trade_pct=Decimal("0.4")))
    assert a != r2.fingerprint(RuntimeSettings(cost_model={"EUR/USD": {"spread": 0.0002}}))
    assert a != r2.fingerprint(_settings())


def _cli_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'e.db'}")
    for k in ("CAPITAL_API_KEY", "BINANCE_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def test_cli_dev_then_one_holdout_and_a_second_look_is_refused(small_round, monkeypatch, tmp_path, capsys):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(r2, "fingerprint", lambda s: "fp-test")
    cli_main(["round2", "--dev"])
    out = capsys.readouterr().out
    assert "CHECK 1:" in out and "Development run number 1" in out
    dev_dir = sorted((tmp_path / "data" / "reports").glob("round2-dev-*"))[-1]
    dev = json.loads((dev_dir / "result.json").read_text())
    assert dev["kind"] == "round2-dev" and dev["fingerprint"] == "fp-test"
    # pretend one strategy passed, so the holdout path runs
    dev["strategies"]["channel_breakout_d1"].update(passed=True, basket=["EUR/USD", "USD/JPY"])
    (tmp_path / "dev.json").write_text(json.dumps(dev))
    cli_main(["round2", "--holdout", str(tmp_path / "dev.json"), "--yes"])
    out = capsys.readouterr().out
    assert "CHECK 2:" in out
    hold = json.loads(sorted((tmp_path / "data" / "reports").glob("round2-holdout-*"))[-1].joinpath("result.json").read_text())
    assert [v["strategy"] for v in hold["verdicts"]] == ["channel_breakout_d1"] and "latest_params" in hold["verdicts"][0]
    with pytest.raises(SystemExit) as e:
        cli_main(["round2", "--holdout", str(tmp_path / "dev.json"), "--yes"])
    assert "Nothing left to run" in str(e.value)
    assert "A second look is not allowed" in capsys.readouterr().out


def test_cli_holdout_refuses_changed_code_or_settings(small_round, monkeypatch, tmp_path):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, tmp_path)
    (tmp_path / "dev.json").write_text(json.dumps({"kind": "round2-dev", "fingerprint": "old", "strategies": {}}))
    with pytest.raises(SystemExit) as e:
        cli_main(["round2", "--holdout", str(tmp_path / "dev.json"), "--yes"])
    assert "changed since that development run" in str(e.value)


def test_cli_holdout_with_nothing_passed_runs_nothing(small_round, monkeypatch, tmp_path):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(r2, "fingerprint", lambda s: "fp")
    (tmp_path / "dev.json").write_text(json.dumps({"kind": "round2-dev", "fingerprint": "fp",
                                                   "strategies": {"tsmom_d1": {"passed": False, "basket": []}}}))
    with pytest.raises(SystemExit) as e:
        cli_main(["round2", "--holdout", str(tmp_path / "dev.json"), "--yes"])
    assert "nothing to run on the holdout" in str(e.value)


def test_dev_run_refuses_an_incomplete_basket(small_round, monkeypatch, tmp_path):
    from tradeengine.cli import main as cli_main

    (small_round / "history" / "dukascopy" / "USDJPY_H1.csv.gz").unlink()
    with pytest.raises(r2.IncompleteData, match="USD/JPY"):
        r2.run_dev(small_round, _settings(), DEFAULT_INSTRUMENTS, progress=lambda *_: None)
    _cli_env(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as e:
        cli_main(["round2", "--dev"])
    assert "Not run: history missing or short for: USD/JPY" in str(e.value)
    assert not (tmp_path / "data" / "reports").exists()  # nothing produced, nothing counted


def test_a_hole_in_the_history_blocks_the_run(small_round):
    from tradeengine.history import read_frame

    path = small_round / "history" / "dukascopy" / "EURUSD_H1.csv.gz"
    df = read_frame(path)
    holed = df[(df.index < pd.Timestamp("2018-03-01", tz="UTC")) | (df.index >= pd.Timestamp("2018-04-01", tz="UTC"))]
    path.unlink()
    write_bars(path, frame_to_bars(holed, "EUR/USD", "H1"))
    with pytest.raises(r2.IncompleteData, match="hole in the data before 2018-04"):
        r2.run_dev(small_round, _settings(), DEFAULT_INSTRUMENTS, progress=lambda *_: None)
