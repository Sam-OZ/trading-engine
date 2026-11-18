"""Round 5: C2's short side on Binance perpetuals. The frozen rules, real funding in the simulator (exact
arithmetic on flat prices), the resumable futures download, the data checks, and the dev -> one-look holdout
order with its fingerprint."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from test_round2 import _cli_env, trending_h1

from tradeengine import round5 as r5
from tradeengine.backtest import cost_for
from tradeengine.history import frame_to_bars, history_path, write_bars
from tradeengine.models import AssetKind, Side, Venue
from tradeengine.portfolio import Leg, Limits, RateBook, simulate
from tradeengine.strategies import Signal, columns


def test_the_pre_registered_settings():
    assert (r5.DEV_FROM, r5.DEV_END, r5.HOLDOUT_END) == (
        pd.Timestamp("2020-03-01", tz="UTC"), pd.Timestamp("2023-11-01", tz="UTC"), pd.Timestamp("2026-11-01", tz="UTC"))
    assert r5.LOOKBACK == {"BTC/USDT-PERP": 14, "ETH/USDT-PERP": 56} and (r5.STOP_ATR, r5.TRAIL_ATR) == (3.0, 4.0)
    assert (r5.RULES.dev_min_trades, r5.RULES.holdout_min_trades, r5.RULES.dev_min_pf, r5.RULES.holdout_min_pf) == (40, 40, 1.2, 1.2)
    assert r5.RULES.bootstrap and r5.RULES.dev_max_dd_pct == r5.RULES.holdout_max_dd_pct == 10.0
    assert cost_for("BTC/USDT-PERP").commission_rate == 0.0005 and cost_for("BTC/USDT").commission_rate == 0.001
    assert r5.plan().lock_prefix == "round5_holdout:"


def test_mirror_trades_one_direction_per_leg():
    d = pd.DataFrame({"close": [100.0, 100.0], "atr": [2.0, 2.0], "ret": [0.05, -0.05]})
    cols, s = columns(d), r5.C2Mirror()
    short, long_ = r5.params_for("BTC/USDT-PERP"), r5.params_for("BTC/USDT-PERP", "long")
    assert s.signal_at(cols, 0, short, True) is None  # rising: the short leg stays out
    sig = s.signal_at(cols, 1, short, True)
    assert sig.side is Side.SELL and sig.stop == 106.0 and sig.trail_atr == 4.0
    assert s.signal_at(cols, 1, short, False) is None  # never short where the market cannot
    assert s.signal_at(cols, 0, long_, False).side is Side.BUY and s.signal_at(cols, 1, long_, True) is None
    assert s.exit_at(cols, 0, Side.SELL, short) and not s.exit_at(cols, 1, Side.SELL, short)


# ------------------------------------------------------------------ funding in the simulator: exact on flat prices
class EnterThenExit:
    name = "probe"

    def __init__(self, side):
        self.side = side

    def signal_at(self, cols, i, p, allow_short):
        if i == 2:
            stop = 150.0 if self.side is Side.SELL else 50.0
            return Signal(self.side, 100.0, stop, None, 1.0, "probe", None, None, None, 0.0)
        return None

    def exit_at(self, cols, i, side, p):
        return i == 6


@pytest.mark.parametrize("side", [Side.SELL, Side.BUY])
def test_funding_is_charged_per_payment_while_held(side):
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    frame = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0, "atr": 1.0}, index=idx)
    rate = 0.0001
    funding = pd.Series(rate, index=pd.date_range("2024-01-01", periods=30, freq="8h", tz="UTC"))
    leg = Leg("BTC/USDT-PERP", AssetKind.CRYPTO_CFD, Venue.BINANCE, "BTC", "USDT", cost_for("BTC/USDT-PERP"), {"k": frame},
              [(idx[0], idx[-1] + pd.Timedelta(days=1), {})], prep_of=lambda p: "k", funding=funding)
    res = simulate(EnterThenExit(side), [leg], RateBook({}), idx[0], idx[-1] + pd.Timedelta(days=1), Limits(), 5000.0)
    (t,) = res.trades
    assert t.entry_time == str(idx[3]) and t.exit_time == str(idx[7])
    held = funding[(funding.index > idx[3]) & (funding.index <= idx[7])]  # 08:00 on entry day ... 00:00 on exit day
    assert len(held) == 12
    o = 100.0  # every bar opens at 100 (spread/slippage only move fills, not the funding notional)
    expected = (1 if side is Side.BUY else -1) * rate * len(held) * t.units * o
    assert t.funding == pytest.approx(expected) and (t.funding < 0) == (side is Side.SELL)  # shorts receive a positive rate
    assert t.costs == pytest.approx(0.0005 * t.units * (t.entry + t.exit) + t.funding)
    assert t.pnl == pytest.approx((t.exit - t.entry) * t.units * (1 if side is Side.BUY else -1) - t.costs)


# ------------------------------------------------------------------ download and data checks
def _klines(start, n):
    out = []
    for k in range(n):
        t = start + pd.Timedelta(hours=k)
        ms = int(t.timestamp() * 1000)
        out.append([ms, "100", "101", "99", "100.5", "10", ms + 3_599_999, "0", 1, "0", "0", "0"])
    return out


@respx.mock
async def test_download_saves_candles_and_funding_and_resumes(tmp_path):
    until = pd.Timestamp("2019-12-01", tz="UTC")
    calls = {"k": 0, "f": 0}

    def klines(request):
        calls["k"] += 1
        st = pd.Timestamp(int(request.url.params["startTime"]), unit="ms", tz="UTC")
        n = min(1500, int((until - st) / pd.Timedelta(hours=1)) + 2)
        return httpx.Response(200, json=_klines(st, max(n, 0)))

    def funding(request):
        calls["f"] += 1
        st = pd.Timestamp(int(request.url.params["startTime"]), unit="ms", tz="UTC").ceil("8h")
        end = pd.Timestamp(int(request.url.params["endTime"]), unit="ms", tz="UTC")
        times = pd.date_range(st, end, freq="8h")[:1000]
        return httpx.Response(200, json=[{"fundingTime": int(t.timestamp() * 1000) + 3, "fundingRate": "0.0001"} for t in times])

    respx.get(f"{r5.FAPI}/fapi/v1/klines").mock(side_effect=klines)
    respx.get(f"{r5.FAPI}/fapi/v1/fundingRate").mock(side_effect=funding)

    async def no_wait(_):
        pass

    notes = await r5.download(tmp_path, until=until, progress=lambda m: None, sleep=no_wait)
    assert set(notes) == set(r5.PERPS)
    h = r5.read_frame(history_path(tmp_path, r5.SOURCE, "BTC/USDT-PERP", "H1"))
    assert h.index[0] == r5.LISTED["BTCUSDT"] and h.index[-1] == until - pd.Timedelta(hours=1)  # only completed candles
    f = r5.read_funding(r5.funding_path(tmp_path, "BTC/USDT-PERP"))
    assert (f.index == f.index.floor("min")).all() and f.iloc[0] == 0.0001  # payment times rounded to the minute
    before = dict(calls)
    await r5.download(tmp_path, until=until, progress=lambda m: None, sleep=no_wait)
    assert calls["k"] == before["k"] and calls["f"] - before["f"] <= 2  # nothing new to fetch


def test_incomplete_data_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="no futures history yet"):
        r5.load_perps(tmp_path)


def _write_funding(data_dir, sym, times):
    p = r5.funding_path(data_dir, sym)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(0.0001, index=times, name="rate").rename_axis("time").to_csv(p)


def test_funding_that_starts_after_listing_but_before_the_test_is_complete(data_dir):
    # Binance's real BTCUSDT record: listed 8 Sep 2019, first funding payment 10 Sep 2019 08:00 (seen 18 Nov 2026).
    end = pd.Timestamp("2026-11-18", tz="UTC")
    _write_funding(data_dir, "BTC/USDT-PERP", pd.date_range(pd.Timestamp("2019-09-10 08:00", tz="UTC"), end, freq="8h"))
    assert set(r5.load_perps(data_dir)) == set(r5.PERPS)

    _write_funding(data_dir, "BTC/USDT-PERP", pd.date_range(pd.Timestamp("2020-03-02", tz="UTC"), end, freq="8h"))
    with pytest.raises(SystemExit, match=r"BTC/USDT-PERP: funding starts 2020-03-02 00:00, after the test starts"):
        r5.load_perps(data_dir)

    times = pd.date_range(pd.Timestamp("2019-09-10 08:00", tz="UTC"), end, freq="8h")
    hole_from, hole_to = pd.Timestamp("2022-05-01", tz="UTC"), pd.Timestamp("2022-05-03", tz="UTC")
    _write_funding(data_dir, "BTC/USDT-PERP", times[(times < hole_from) | (times > hole_to)])
    with pytest.raises(SystemExit, match="gap in the funding payments"):
        r5.load_perps(data_dir)


# ------------------------------------------------------------------ the two looks, in order
@pytest.fixture(scope="module")
def market():
    return {
        "BTC/USDT-PERP": trending_h1("2019-12-01", "2026-11-01", seed=51, base=8000.0, crypto=True, vol=0.006, drift=0.0, regime_hours=24 * 20),
        "ETH/USDT-PERP": trending_h1("2019-12-01", "2026-11-01", seed=52, base=150.0, crypto=True, vol=0.007, drift=0.0, regime_hours=24 * 20),
    }


@pytest.fixture
def data_dir(market, tmp_path):
    rng = np.random.default_rng(5)
    for sym, df in market.items():
        write_bars(history_path(tmp_path, r5.SOURCE, sym, "H1"), frame_to_bars(df, sym, "H1"))
        write_bars(history_path(tmp_path, "binance", r5.PERPS[sym][1], "H1"), frame_to_bars(df, r5.PERPS[sym][1], "H1"))
        times = pd.date_range(r5.LISTED[r5.PERPS[sym][0]], pd.Timestamp("2026-11-01", tz="UTC"), freq="8h")
        p = r5.funding_path(tmp_path, sym)
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.Series(rng.normal(0.0001, 0.0002, len(times)), index=times, name="rate").rename_axis("time").to_csv(p)
    return tmp_path


def test_holdout_needs_a_dev_pass_with_the_same_fingerprint_and_is_one_look(data_dir, monkeypatch, capsys):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, data_dir)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    with pytest.raises(SystemExit, match="needs a development PASS"):
        cli_main(["round5", "--holdout", "--yes"])
    real_judge = r5.judge
    monkeypatch.setattr(r5, "judge", lambda m, which: (True, []) if which == "dev" else real_judge(m, which))
    cli_main(["round5", "--dev"])
    out = capsys.readouterr().out
    assert "Round 5 (development)" in out and "**Verdict: PASS**" in out
    rep = sorted((data_dir / "reports").glob("round5-dev-*"))[-1]
    res = json.loads((rep / "result.json").read_text())
    trades = pd.read_csv(rep / "trades.csv")
    assert res["metrics"]["trades"] > 0 and (trades["side"] == "SELL").all()  # short only
    assert (pd.to_datetime(trades["entry_time"], utc=True) < r5.DEV_END).all()
    assert trades["funding"].abs().sum() > 0  # real funding was applied

    cli_main(["round5", "--holdout", "--yes"])
    out = capsys.readouterr().out
    assert "Round 5 (unseen, one look)" in out and "C2 long (spot) + C2-short together, information only" in out
    rep = sorted((data_dir / "reports").glob("round5-holdout-*"))[-1]
    trades = pd.read_csv(rep / "trades.csv")
    assert (pd.to_datetime(trades["entry_time"], utc=True) >= r5.DEV_END).all()
    with pytest.raises(SystemExit, match="already taken"):
        cli_main(["round5", "--holdout", "--yes"])


def test_a_changed_cost_after_the_dev_pass_blocks_the_unseen_look(data_dir, monkeypatch):
    from tradeengine.cli import main as cli_main

    _cli_env(monkeypatch, data_dir)
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setattr(r5, "judge", lambda m, which: (True, []))
    cli_main(["round5", "--dev"])
    cli_main(["settings", "--set", 'cost_model={"BTC/USDT-PERP": {"commission_rate": 0.0002}}'])
    with pytest.raises(SystemExit, match="changed since the development PASS"):
        cli_main(["round5", "--holdout", "--yes"])
