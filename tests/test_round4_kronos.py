"""Round 4 Track B (Kronos): the frozen rules, no look-ahead in the forecasts, a resumable cache, and the
one-look score against C2 over the same window. The real model is replaced by a fake forecaster; the real
one needs torch and runs on Sam's Mac."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from test_round2 import _cli_env, trending_h1

from tradeengine import round4_kronos as r4k
from tradeengine.history import frame_to_bars, history_path, write_bars
from tradeengine.models import Side
from tradeengine.strategies import columns

START = "2023-11-01"


@pytest.fixture(scope="module")
def h1():
    return {
        "BTC/USDT": trending_h1(START, "2026-11-01", seed=41, base=30000.0, crypto=True, vol=0.006, drift=0.0006, regime_hours=24 * 30),
        "ETH/USDT": trending_h1(START, "2026-11-01", seed=42, base=2000.0, crypto=True, vol=0.007, drift=0.0006, regime_hours=24 * 25),
    }


@pytest.fixture
def data_dir(h1, tmp_path, monkeypatch):
    monkeypatch.setattr(r4k, "DATA_FROM", pd.Timestamp(START, tz="UTC"))
    for sym, df in h1.items():
        write_bars(history_path(tmp_path, "binance", sym, "H1"), frame_to_bars(df, sym, "H1"))
    return tmp_path


class FakeForecaster:
    """Forecasts the 14-day momentum forward; records exactly what it was shown."""

    def __init__(self):
        self.calls = []

    def forecast(self, frames, xs, ys, seed):
        self.calls.append((frames, xs, ys, seed))
        out = []
        for f, y in zip(frames, ys, strict=True):
            mom = f["close"].iloc[-1] / f["close"].iloc[-15] - 1
            out.append(pd.DataFrame({"close": f["close"].iloc[-1] * (1 + mom) * np.ones(len(y))}, index=y))
        return out


def daily_of(h1):
    return {s: r4k.daily_frame(df) for s, df in h1.items()}


def test_the_pre_registered_settings():
    assert r4k.SYMBOLS == ["BTC/USDT", "ETH/USDT"]
    assert (r4k.SCORE_FROM, r4k.SCORE_END) == (pd.Timestamp("2025-08-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
    assert (r4k.MODEL_ID, r4k.TOKENIZER_ID) == ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base")
    assert (r4k.CONTEXT, r4k.HORIZON, r4k.SAMPLES, r4k.TOP_P) == (512, 7, 20, 0.9)
    assert r4k.RULES.holdout_min_trades == 30 and r4k.RULES.holdout_min_pf == 1.2 and r4k.RULES.bootstrap
    assert r4k.C2_PARAMS["BTC/USDT"]["lookback"] == 14 and r4k.C2_PARAMS["ETH/USDT"]["lookback"] == 56
    assert r4k.plan().lock_prefix == "round4_holdout:"
    assert r4k.round_trip_cost("BTC/USDT") == pytest.approx(2 * (0.001 + 1 / 20000 + 2 / 10000))


def test_strategy_buys_only_above_costs_and_exits_when_the_forecast_turns(h1):
    d = r4k.daily_frame(h1["BTC/USDT"])
    d["fret"] = np.linspace(-0.05, 0.05, len(d))
    cols = columns(d)
    s, p = r4k.KronosD1(), {"min_ret": 0.005, "stop_atr": 3.0, "trail_atr": 4.0}
    sigs = [(i, s.signal_at(cols, i, p, True)) for i in range(25, len(d))]
    buys = [(i, x) for i, x in sigs if x is not None]
    assert buys and all(x.side is Side.BUY and d["fret"].iloc[i] > 0.005 for i, x in buys)  # long only, even if shorts allowed
    i, x = buys[0]
    assert x.stop == pytest.approx(d["close"].iloc[i] - 3.0 * d["atr"].iloc[i]) and x.trail_atr == 4.0 and x.trail_activate_r == 0.0
    exits = [i for i in range(len(d)) if s.exit_at(cols, i, Side.BUY, p)]
    assert exits and all(d["fret"].iloc[i] <= 0 for i in exits)


def test_forecasts_see_only_completed_days_and_the_cache_resumes(h1, tmp_path):
    daily = daily_of(h1)
    path = tmp_path / "f.csv"
    fake = FakeForecaster()
    assert r4k.make_forecasts(daily, fake, path, progress=lambda m: None, limit=5) == 5
    for frames, xs, ys, seed in fake.calls:
        day = xs[0].iloc[-1]
        assert all(len(f) == r4k.CONTEXT for f in frames)  # 512 completed daily candles, the last one is the day itself
        assert all(x.iloc[-1] == day for x in xs) and all(y.iloc[0] == day + pd.Timedelta(days=1) for y in ys)
        assert all(len(y) == r4k.HORIZON for y in ys) and seed == r4k.SEED + int(day.timestamp() // 86400)
        btc = daily["BTC/USDT"]
        assert frames[0]["close"].iloc[-1] == btc.loc[day, "close"]  # nothing after the day's close
    first = fake.calls[0][1][0].iloc[-1]
    assert first == r4k.SCORE_FROM
    rest = r4k.make_forecasts(daily, fake, path, progress=lambda m: None)
    days = sorted(set(r4k.scoring_days(daily["BTC/USDT"])) | set(r4k.scoring_days(daily["ETH/USDT"])))
    assert 5 + rest == len(days) and r4k.make_forecasts(daily, fake, path, progress=lambda m: None) == 0
    cache = r4k.load_cache(path)
    assert len(cache) == 2 * len(days) and not cache.duplicated(["symbol", "open_time"]).any()
    assert r4k.missing_forecasts(daily, cache) == {"BTC/USDT": 0, "ETH/USDT": 0}
    assert days[-1] == pd.Timestamp("2026-10-31", tz="UTC")  # the last daily close scored is 1 Nov 00:00 UTC


def test_forecast_without_the_optional_packages_says_how_to_install_them():
    try:
        import torch  # noqa: F401
    except ImportError:
        with pytest.raises(SystemExit) as e:
            r4k.Forecaster()
        assert "--extra kronos" in str(e.value)
    else:
        pytest.skip("torch is installed here")


def test_cli_score_is_one_look_and_compares_with_c2(data_dir, h1, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, data_dir)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    with pytest.raises(SystemExit) as e:  # no forecasts yet: refused
        cli_main(["round4-kronos", "--score", "--yes"])
    assert "Forecasts missing" in str(e.value)

    monkeypatch.setattr(r4k, "Forecaster", lambda device=None: type("F", (FakeForecaster,), {"device": "fake"})())
    cli_main(["round4-kronos", "--forecast"])
    assert "Every forecast is ready" in capsys.readouterr().out

    cli_main(["round4-kronos", "--score", "--yes"])
    out = capsys.readouterr().out
    assert "Round 4 Track B: Kronos" in out and "C2 (tsmom_fast_d1), same window" in out
    report = sorted((data_dir / "reports").glob("round4-kronos-*"))[-1]
    res = json.loads((report / "result.json").read_text())
    assert res["verdict"] in ("PASS", "FAIL") and res["kronos"]["trades"] > 0 and res["c2"]["trades"] > 0
    assert (report / "kronos_trades.csv").exists()
    trades = pd.read_csv(report / "kronos_trades.csv")
    assert (pd.to_datetime(trades["entry_time"], utc=True) >= r4k.SCORE_FROM).all()  # nothing traded before the window

    with pytest.raises(SystemExit) as e:  # the second look is refused
        cli_main(["round4-kronos", "--score", "--yes"])
    assert "already scored" in str(e.value)


def test_judge_needs_every_mark_and_must_beat_c2():
    base = {"trades": 40, "profit_factor": 1.5, "max_drawdown_pct": 5.0, "bootstrap_lower_bound": 0.02}
    ok, why = r4k.judge(base, {**base, "profit_factor": 1.4})
    assert ok and why == []
    ok, why = r4k.judge(base, {**base, "profit_factor": 1.6})
    assert not ok and "does not beat C2" in why[0]
    ok, why = r4k.judge({**base, "trades": 20, "bootstrap_lower_bound": -0.01}, {**base, "profit_factor": 1.0})
    assert not ok and len(why) == 2
