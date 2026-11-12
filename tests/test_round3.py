"""Round 3 (crypto): the fast momentum strategy, the statistical check, and the round 3 commands with
their own one-look lock."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_round2 import _cli_env, trending_h1

from tradeengine import round2 as r2
from tradeengine import round3 as r3
from tradeengine.backtest import Trade
from tradeengine.history import frame_to_bars, history_path, write_bars
from tradeengine.models import DEFAULT_INSTRUMENTS, Side
from tradeengine.settings_store import RuntimeSettings
from tradeengine.strategies import columns
from tradeengine.trend import ROUND3_STRATEGIES, TimeSeriesMomentumFastD1, prepare


@pytest.fixture(scope="module")
def btc():
    return trending_h1("2019-01-01", "2021-01-01", seed=31, base=8000.0, crypto=True, vol=0.006, drift=0.0006, regime_hours=24 * 30)


def test_round3_is_btc_and_eth_with_the_two_planned_strategies():
    pl = r3.plan()
    assert pl.symbols == ["BTC/USDT", "ETH/USDT"] and list(pl.strategies) == ["breakout_h4_dfilter", "tsmom_fast_d1"]
    assert pl.rules.bootstrap and pl.rules.dev_min_trades == 60 and pl.rules.holdout_min_trades == 40
    assert pl.rules.basket_min_trades == 15 and pl.lock_prefix == "round3_holdout:"
    assert ROUND3_STRATEGIES["tsmom_fast_d1"].grid["lookback"] == [14, 28, 56]
    assert ROUND3_STRATEGIES["breakout_h4_dfilter"].grid == {"dc_n": [20, 55], "stop_atr": [2.0, 3.0], "trail_atr": [3.0]}
    assert RuntimeSettings().round3_walk_forward == {"breakout_h4_dfilter": [24, 6], "tsmom_fast_d1": [24, 6]}


def test_fast_momentum_checks_every_day_and_exits_when_the_return_is_no_longer_positive(btc):
    s = TimeSeriesMomentumFastD1()
    p = {"lookback": 14, "stop_atr": 3.0, "trail_atr": 4.0}
    f = s.prepare(btc, p, crypto=True)
    cols = columns(f)
    buys = [i for i in range(len(f)) if s.signal_at(cols, i, p, False) is not None]
    assert len({int(f["weekday"].iloc[i]) for i in buys}) == 7  # any day of the week, not only Fridays
    assert all(f["ret"].iloc[i] > 0 for i in buys)
    assert all(s.signal_at(cols, i, p, False) is None or s.signal_at(cols, i, p, False).side is Side.BUY for i in range(len(f)))
    exits = [i for i in range(len(f)) if s.exit_at(cols, i, Side.BUY, p)]
    assert exits and all(f["ret"].iloc[i] <= 0 for i in exits)


@pytest.mark.parametrize("name", list(ROUND3_STRATEGIES))
def test_round3_signals_identical_with_or_without_future_data(btc, name):
    s = ROUND3_STRATEGIES[name]
    p = s.default_params
    full = prepare(s, btc, p, True)
    cols = columns(full)
    ready = np.flatnonzero(full["ready"].to_numpy(dtype=bool))
    for i in ready[:: max(1, len(ready) // 20)][:20]:
        cut = btc[btc.index < pd.Timestamp(full["closes_at"].iloc[i])]
        part = prepare(s, cut, p, True)
        assert part.index[-1] == full.index[i]
        a, b = s.signal_at(cols, i, p, False), s.signal_at(columns(part), len(part) - 1, p, False)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.stop == pytest.approx(b.stop)


def test_bootstrap_lower_bound_is_seeded_and_separates_an_edge_from_noise():
    rules = r3.ROUND3_RULES
    rng = np.random.default_rng(1)
    edge = list(rng.normal(0.4, 1.0, 120))
    noise = list(rng.normal(0.0, 1.0, 120))
    assert r2.bootstrap_lower_bound(edge, rules) == r2.bootstrap_lower_bound(edge, rules)  # same trades, same number
    assert r2.bootstrap_lower_bound(edge, rules) > 0
    assert r2.bootstrap_lower_bound(noise, rules) < 0  # no edge: does not clear the bar
    assert r2.bootstrap_lower_bound([1.0], rules) == float("-inf")
    # fewer trades need a larger edge: the same average with a quarter of the trades has a lower bound
    assert r2.bootstrap_lower_bound(edge[:30], rules) < r2.bootstrap_lower_bound(edge, rules)


def _run(rs: list[float]) -> r2.StrategyRun:
    from tradeengine.portfolio import PortfolioMetrics

    trades = [Trade("BTC/USDT", "x", "p", "BUY", "2020-01-01", "2020-01-02", 1, 1, 1, None, 1, 1, r, r, 0, 1, "stop") for r in rs]
    w, lo = sum(r for r in rs if r > 0), -sum(r for r in rs if r <= 0)
    m = PortfolioMetrics(len(rs), 0.5, w / lo, float(np.mean(rs)), float(sum(rs)), 5.0, {"binance": 5.0}, 10.0, 1.0, 30.0)
    return r2.StrategyRun("x", m, {"BTC/USDT": (len(rs), w / lo, sum(rs))}, {}, {}, None, trades, pd.DataFrame(), 7, ("a", "b"))


def test_the_statistical_check_can_fail_a_strategy_that_meets_the_other_bars():
    rules = r3.ROUND3_RULES
    lucky = [3.0, -1.0, -1.0, 2.5, -1.0, -1.0] * 11  # 66 trades, PF 1.38, but too few and too uneven to be sure
    ok, reasons, basket = r2.judge_dev(_run(lucky), rules)
    assert not ok and any("statistical check" in r for r in reasons) and not any("trades, needs" in r for r in reasons)
    solid = [1.5, -1.0, 1.2, -0.8, 1.4, 0.9] * 12
    ok, reasons, basket = r2.judge_dev(_run(solid), rules)
    assert ok and basket == ["BTC/USDT"]
    # round 2's rules have no statistical check: unchanged behaviour
    assert not any("statistical" in r for r in r2.judge_dev(_run(lucky), r2.RULES)[1])


def test_fingerprints_differ_between_rounds():
    s = RuntimeSettings()
    assert r2.fingerprint(s, r2.plan()) != r2.fingerprint(s, r3.plan())
    assert r2.fingerprint(s, r3.plan()) != r2.fingerprint(RuntimeSettings(round3_walk_forward={"tsmom_fast_d1": [12, 6]}), r3.plan())


@pytest.fixture
def small_crypto_round(monkeypatch, tmp_path):
    monkeypatch.setattr(r3, "ROUND3_DEV_END", pd.Timestamp("2021-01-01", tz="UTC"))
    monkeypatch.setattr(r3, "ROUND3_HOLDOUT_END", pd.Timestamp("2022-01-01", tz="UTC"))
    for strat in ROUND3_STRATEGIES.values():
        monkeypatch.setattr(strat, "min_is_trades", 2)
    data = tmp_path / "data"
    for k, (sym, px) in enumerate([("BTC/USDT", 9000.0), ("ETH/USDT", 300.0)]):
        h1 = trending_h1("2018-01-01", "2022-02-01", seed=40 + k, base=px, crypto=True, vol=0.005, drift=0.0004, regime_hours=24 * 40)
        write_bars(history_path(data, "binance", sym, "H1"), frame_to_bars(h1, sym, "H1"))
    return data


def _settings():
    return RuntimeSettings(round3_walk_forward={"breakout_h4_dfilter": [12, 6], "tsmom_fast_d1": [12, 6]})


def test_round3_dev_run_is_crypto_only_long_only_and_reports_the_statistical_check(small_crypto_round):
    pl = r3.plan()
    runs, verdicts, notes = r2.run_dev(small_crypto_round, _settings(), DEFAULT_INSTRUMENTS, progress=lambda *_: None, pl=pl)
    assert [r.strategy for r in runs] == ["breakout_h4_dfilter", "tsmom_fast_d1"]
    for run in runs:
        assert run.trades and {t.symbol for t in run.trades} <= {"BTC/USDT", "ETH/USDT"}
        assert all(t.side == "BUY" for t in run.trades) and all(pd.Timestamp(t.exit_time) < pl.dev_end for t in run.trades)
        assert verdicts[run.strategy]["lower_bound"] is not None
    md = r2.render("dev", runs, verdicts, notes, "fp", [], None, pl)
    assert md.startswith("# Round 3") and "Lower bound R" in md and "statistical check" in md


def test_cli_round3_has_its_own_one_look_lock(small_crypto_round, monkeypatch, tmp_path, capsys):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, tmp_path)
    monkeypatch.setattr(r2, "fingerprint", lambda *a: "fp3")

    async def fake_load(self):
        return _settings()

    from tradeengine.settings_store import SettingsStore

    monkeypatch.setattr(SettingsStore, "load", fake_load)
    cli_main(["round3", "--dev"])
    out = capsys.readouterr().out
    assert "# Round 3" in out and "CHECK 1:" in out
    dev_dir = sorted((tmp_path / "data" / "reports").glob("round3-dev-*"))[-1]
    dev = json.loads((dev_dir / "result.json").read_text())
    assert dev["kind"] == "round3-dev"
    dev["strategies"]["tsmom_fast_d1"].update(passed=True, basket=["BTC/USDT", "ETH/USDT"])
    (tmp_path / "dev3.json").write_text(json.dumps(dev))
    # a round 2 result is not accepted by round 3
    (tmp_path / "dev2.json").write_text(json.dumps({**dev, "kind": "round2-dev"}))
    with pytest.raises(SystemExit) as e:
        cli_main(["round3", "--holdout", str(tmp_path / "dev2.json"), "--yes"])
    assert "not a round 3 development result" in str(e.value)
    cli_main(["round3", "--holdout", str(tmp_path / "dev3.json"), "--yes"])
    out = capsys.readouterr().out
    assert "# Round 3 — HOLDOUT" in out and "CHECK 2:" in out
    hold = json.loads(sorted((tmp_path / "data" / "reports").glob("round3-holdout-*"))[-1].joinpath("result.json").read_text())
    assert hold["kind"] == "round3-holdout" and [v["strategy"] for v in hold["verdicts"]] == ["tsmom_fast_d1"]
    with pytest.raises(SystemExit) as e:
        cli_main(["round3", "--holdout", str(tmp_path / "dev3.json"), "--yes"])
    assert "Nothing left to run" in str(e.value)
