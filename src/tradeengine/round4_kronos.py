"""Round 4, Track B: a fair test of the Kronos forecasting model. Approved by Sam 17 Nov 2026
(plan: Claude Docs "Round 4 plan - follow good traders and test Kronos"). The constants below ARE the
pre-registration: nothing here is tuned after the scored run.

- Bitcoin and Ethereum spot on Binance, long only, the same costs as round 3.
- Kronos-small with Kronos-Tokenizer-base (Hugging Face NeoQuasar), the last 512 daily candles as context,
  7 daily candles forecast, 20 samples averaged (Kronos averages them itself), T 1.0, top-p 0.9, seeded per day.
- At each daily close (00:00 UTC): forecast return = forecast close in 7 days / today's close - 1.
  Buy when it is above twice the coin's round-trip cost; exit at the first daily close where it is <= 0.
  Stop 3 x ATR(20), trailed 4 x ATR from the start, 0.5% risk, every portfolio limit: exactly C2's.
- Scored ONLY on daily closes from 1 Aug 2025 to 31 Oct 2026 (Kronos was published in Aug 2025; earlier
  prices may be in its training data, so testing on them would flatter it).
- Pass: >= 30 trades, profit factor >= 1.2, drawdown < 10%, 5th percentile of 10,000 seeded bootstrap means
  of trade R > 0, AND a higher profit factor than C2 over the same window.
- One look: the score is locked in system_state (round4_holdout:kronos_d1) with a fingerprint.

Two steps, both on a desktop (the server never installs torch):
  uv run --extra kronos tradeengine round4-kronos --forecast   # resumable; caches every forecast
  uv run tradeengine round4-kronos --score                     # one look, once every forecast exists
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import indicators as ind
from . import portfolio as pf_mod
from . import trend as trend_mod
from .backtest import cost_for
from .models import Side
from .portfolio import Leg, Limits, portfolio_metrics, simulate, symbol_stats
from .round2 import (
    START_EQUITY,
    Plan,
    Rules,
    bootstrap_lower_bound,
    load_round2_data,
    rate_book,
    require_complete,
)
from .strategies import Signal
from .trend import TimeSeriesMomentumFastD1, _finite, daily_from_h1, prep_key, prepare

SYMBOLS = ["BTC/USDT", "ETH/USDT"]
DATA_FROM = pd.Timestamp("2018-01-01", tz="UTC")
SCORE_FROM = pd.Timestamp("2025-08-01", tz="UTC")
SCORE_END = pd.Timestamp("2026-11-01", tz="UTC")  # daily bars opening before this (last close 1 Nov 00:00 UTC)

MODEL_ID = "NeoQuasar/Kronos-small"
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
CONTEXT = 512
HORIZON = 7
SAMPLES = 20
TEMPERATURE = 1.0
TOP_P = 0.9
SEED = 20261117
COST_MULTIPLE = 2.0  # buy only when the forecast gain beats twice the round-trip cost
STOP_ATR, TRAIL_ATR = 3.0, 4.0
C2_PARAMS = {"BTC/USDT": {"lookback": 14, "stop_atr": 3.0, "trail_atr": 4.0},
             "ETH/USDT": {"lookback": 56, "stop_atr": 3.0, "trail_atr": 4.0}}  # the forward test's settings

RULES = Rules(holdout_min_pf=1.2, holdout_max_dd_pct=10.0, holdout_min_trades=30, bootstrap=True)
STRATEGY = "kronos_d1"


def plan() -> Plan:
    return Plan(4, SYMBOLS, {"crypto": DATA_FROM, "fx": DATA_FROM}, SCORE_FROM, SCORE_END, RULES, {}, "round3_walk_forward")


def config() -> dict:
    return {"model": MODEL_ID, "tokenizer": TOKENIZER_ID, "context": CONTEXT, "horizon": HORIZON, "samples": SAMPLES,
            "temperature": TEMPERATURE, "top_p": TOP_P, "seed": SEED, "score_from": str(SCORE_FROM), "score_end": str(SCORE_END)}


def config_key() -> str:
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()[:12]


def round_trip_cost(symbol: str, overrides: dict | None = None) -> float:
    """Fraction of price lost entering and leaving once: fees, half the spread and slippage, both ways."""
    c = cost_for(symbol, overrides)
    return 2 * (c.commission_rate + c.spread_bps / 20000 + c.slippage_bps / 10000)


# ------------------------------------------------------------------ the strategy
@dataclass
class KronosD1:
    """Long while Kronos's 7-day forecast beats costs; out at the first close where it is not positive."""

    name: str = STRATEGY
    timeframe: str = "D1"

    def signal_at(self, cols, i, p, allow_short):
        c, atr, fret = cols["close"][i], cols["atr"][i], cols["fret"][i]
        if not _finite(c, atr, fret) or atr <= 0:
            return None
        if fret > p["min_ret"]:
            return Signal(Side.BUY, c, c - p["stop_atr"] * atr, None, atr, f"Kronos 7-day forecast {fret:+.1%}",
                          p["trail_atr"], None, None, 0.0)
        return None  # long only: a negative forecast means stay out

    def exit_at(self, cols, i, side, p) -> bool:
        fret = cols["fret"][i]
        return _finite(fret) and side is Side.BUY and fret <= 0


def daily_frame(h1: pd.DataFrame) -> pd.DataFrame:
    d = daily_from_h1(h1, crypto=True)
    d["atr"] = ind.atr(d["high"], d["low"], d["close"], 20)
    return d


def scoring_days(daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Daily bars whose close is scored: open in [SCORE_FROM, SCORE_END), with CONTEXT bars of history."""
    idx = daily.index
    pos = np.arange(len(idx))
    keep = (idx >= SCORE_FROM) & (idx < SCORE_END) & (pos >= CONTEXT - 1)
    return idx[keep]


# ------------------------------------------------------------------ forecasts (cached, resumable)
def cache_path(data_dir: Path) -> Path:
    return data_dir / "kronos" / f"forecasts-{config_key()}.csv"


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "open_time", "close", "fclose7", "fret"])
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


class Forecaster:
    """The real model. Imported only here, so the rest of the engine never needs torch."""

    def __init__(self, device: str | None = None):
        try:
            import torch  # noqa: F401

            from .vendor.kronos import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as e:
            raise SystemExit(f"Kronos needs the optional packages: run with `uv run --extra kronos ...` ({e}).") from e
        tok = KronosTokenizer.from_pretrained(TOKENIZER_ID)
        model = Kronos.from_pretrained(MODEL_ID)
        tok.eval()
        model.eval()
        self.predictor = KronosPredictor(model, tok, device=device, max_context=CONTEXT)
        self.device = self.predictor.device

    def forecast(self, frames: list[pd.DataFrame], x_ts: list[pd.Series], y_ts: list[pd.Series], seed: int) -> list[pd.DataFrame]:
        import torch

        torch.manual_seed(seed)
        return self.predictor.predict_batch(frames, x_ts, y_ts, pred_len=HORIZON, T=TEMPERATURE, top_k=0, top_p=TOP_P,
                                            sample_count=SAMPLES, verbose=False)


def make_forecasts(daily: dict[str, pd.DataFrame], forecaster, path: Path, progress: Callable[[str], None] = print,
                   limit: int | None = None) -> int:
    """Forecast every scoring day not yet in the cache; both coins in one batch per day. Returns days added."""
    cache = load_cache(path)
    done = {(r.symbol, r.open_time) for r in cache.itertuples()}
    days = sorted(set().union(*(set(scoring_days(daily[s])) for s in SYMBOLS)))
    todo = [d for d in days if any((s, d) not in done for s in SYMBOLS if d in daily[s].index)]
    progress(f"Kronos forecasts: {len(days) - len(todo)} of {len(days)} days cached, {len(todo)} to go ({path})")
    if limit is not None:
        todo = todo[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / f"forecasts-{config_key()}.json").write_text(json.dumps(config(), indent=2))
    started = datetime.now(UTC)
    for n, day in enumerate(todo, 1):
        syms = [s for s in SYMBOLS if day in daily[s].index and (s, day) not in done]
        frames, xs, ys = [], [], []
        for s in syms:
            d = daily[s]
            i = d.index.get_loc(day)
            ctx = d.iloc[i - CONTEXT + 1: i + 1]
            frames.append(ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
            xs.append(pd.Series(ctx.index))
            ys.append(pd.Series(pd.date_range(day + pd.Timedelta(days=1), periods=HORIZON, freq="D")))
        seed = SEED + int(day.timestamp() // 86400)
        preds = forecaster.forecast(frames, xs, ys, seed)
        rows = []
        for s, ctx, pred in zip(syms, frames, preds, strict=True):
            close = float(ctx["close"].iloc[-1])
            f7 = float(pred["close"].iloc[HORIZON - 1])
            rows.append({"symbol": s, "open_time": day.isoformat(), "close": close, "fclose7": f7, "fret": f7 / close - 1})
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)
        if n == 1 or n % 10 == 0 or n == len(todo):
            per = (datetime.now(UTC) - started).total_seconds() / n
            progress(f"  {day:%Y-%m-%d} done ({n}/{len(todo)}), about {per * (len(todo) - n) / 60:.0f} min left")
    return len(todo)


def attach(daily: pd.DataFrame, cache: pd.DataFrame, symbol: str) -> pd.DataFrame:
    f = cache[cache["symbol"] == symbol].set_index("open_time")["fret"]
    out = daily.copy()
    out["fret"] = f.reindex(out.index)
    return out


def missing_forecasts(daily: dict[str, pd.DataFrame], cache: pd.DataFrame) -> dict[str, int]:
    have = {(r.symbol, r.open_time) for r in cache.itertuples()}
    return {s: sum((s, d) not in have for d in scoring_days(daily[s])) for s in SYMBOLS}


# ------------------------------------------------------------------ scoring (one look)
@dataclass
class Verdict:
    passed: bool
    reasons: list[str]
    kronos: dict
    c2: dict


def _summary(res, rules: Rules) -> dict:
    m = portfolio_metrics(res)
    d = asdict(m)
    d["profit_factor"] = None if math.isinf(m.profit_factor) else round(m.profit_factor, 3)
    d["bootstrap_lower_bound"] = round(bootstrap_lower_bound([t.r for t in res.trades], rules), 4) if res.trades else None
    d["per_symbol"] = {s: [n, (None if math.isinf(p) else round(p, 3)), round(r, 2)] for s, (n, p, r) in symbol_stats(res.trades).items()}
    d["halted_at"] = res.halted_at
    d["rejections"] = dict(res.rejections)
    return d


def run_both(data: dict, settings, cache: pd.DataFrame) -> tuple:
    """Kronos and C2 on the same coins, window, costs and limits. Returns (kronos result, c2 result)."""
    rates = rate_book(data)
    limits = Limits.from_settings(settings)
    kronos, c2 = KronosD1(), TimeSeriesMomentumFastD1()
    k_legs, c_legs = [], []
    for sym in SYMBOLS:
        h1, inst = data[sym]
        cost = cost_for(sym, settings.cost_model)
        frame = attach(daily_frame(h1), cache, sym)
        p = {"min_ret": COST_MULTIPLE * round_trip_cost(sym, settings.cost_model), "stop_atr": STOP_ATR, "trail_atr": TRAIL_ATR}
        k_legs.append(Leg(sym, inst.kind, inst.venue, inst.base, inst.quote, cost, {"kronos": frame}, [(SCORE_FROM, SCORE_END, p)],
                          prep_of=lambda q: "kronos"))
        cp = C2_PARAMS[sym]
        c_legs.append(Leg(sym, inst.kind, inst.venue, inst.base, inst.quote, cost, {prep_key(c2, cp): prepare(c2, h1, cp, True)},
                          [(SCORE_FROM, SCORE_END, cp)], prep_of=lambda q, s=c2: prep_key(s, q)))
    kr = simulate(kronos, k_legs, rates, SCORE_FROM, SCORE_END, limits, START_EQUITY)
    cr = simulate(c2, c_legs, rates, SCORE_FROM, SCORE_END, limits, START_EQUITY)
    return kr, cr


def judge(k: dict, c: dict, rules: Rules = RULES) -> tuple[bool, list[str]]:
    reasons = []
    if k["trades"] < rules.holdout_min_trades:
        reasons.append(f"{k['trades']} trades, needs {rules.holdout_min_trades}")
    kpf = math.inf if k["profit_factor"] is None and k["trades"] else (k["profit_factor"] or 0.0)
    if kpf < rules.holdout_min_pf:
        reasons.append(f"profit factor {kpf:.2f}, needs {rules.holdout_min_pf}")
    if k["max_drawdown_pct"] >= rules.holdout_max_dd_pct:
        reasons.append(f"drawdown {k['max_drawdown_pct']:.1f}%, limit {rules.holdout_max_dd_pct}%")
    lb = k["bootstrap_lower_bound"]
    if lb is None or lb <= 0:
        reasons.append(f"statistical check: 5th-percentile average R {lb if lb is not None else 'n/a'}, needs > 0")
    cpf = math.inf if c["profit_factor"] is None and c["trades"] else (c["profit_factor"] or 0.0)
    if not kpf > cpf:
        reasons.append(f"profit factor {kpf:.2f} does not beat C2's {cpf:.2f} over the same window")
    return not reasons, reasons


def fingerprint(settings, cache_file: Path) -> str:
    from . import round4_kronos as me

    parts = [inspect.getsource(m) for m in (me, pf_mod, trend_mod)]
    vendor = Path(__file__).parent / "vendor" / "kronos"
    parts += [(vendor / f).read_text() for f in ("kronos.py", "module.py")]
    parts.append(hashlib.sha256(cache_file.read_bytes()).hexdigest())
    parts.append(json.dumps({s: asdict(cost_for(s, settings.cost_model)) for s in SYMBOLS}, sort_keys=True))
    parts.append(json.dumps(asdict(Limits.from_settings(settings)), sort_keys=True))
    parts.append(json.dumps({"config": config(), "rules": asdict(RULES), "c2": C2_PARAMS}, sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def load_data(data_dir: Path, instruments: dict) -> tuple[dict, list[str]]:
    data, notes = load_round2_data(data_dir, instruments, SCORE_END, plan())
    require_complete(notes, data)
    for sym, (h1, _) in data.items():
        if h1.index[-1] < SCORE_END - pd.Timedelta(hours=2):
            raise SystemExit(f"{sym}: hourly history ends {h1.index[-1]:%Y-%m-%d %H:%M}, before {SCORE_END:%Y-%m-%d}. "
                             "Run `./scripts/with-secrets.sh uv run tradeengine download-all --from 2013-01-01` first.")
    return data, notes


def render(v: Verdict, fp: str, notes: list[str], cache_file: Path) -> str:
    def row(name, m):
        pf = "inf" if m["profit_factor"] is None and m["trades"] else f"{(m['profit_factor'] or 0):.2f}"
        lb = "n/a" if m["bootstrap_lower_bound"] is None else f"{m['bootstrap_lower_bound']:+.3f}"
        return (f"| {name} | {m['trades']} | {m['win_rate']:.0%} | {pf} | {m['expectancy_r']:+.2f} | "
                f"{m['max_drawdown_pct']:.1f}% | {m['return_pct']:+.1f}% | {lb} |")

    lines = [
        f"# Round 4 Track B: Kronos, scored {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        f"**Verdict: {'PASS' if v.passed else 'FAIL'}**" + ("" if v.passed else " - " + "; ".join(v.reasons)),
        "",
        f"Window: daily closes {SCORE_FROM:%d %b %Y} to {SCORE_END - pd.Timedelta(days=1):%d %b %Y}; BTC/USDT and ETH/USDT, long only, "
        "0.5% risk, round 3 costs, every portfolio limit.",
        "",
        "| Strategy | Trades | Win | Profit factor | Avg R | Max drawdown | Return | Bootstrap low bound |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        row("Kronos (kronos_d1)", v.kronos),
        row("C2 (tsmom_fast_d1), same window", v.c2),
        "",
        f"Model: {MODEL_ID} + {TOKENIZER_ID}; context {CONTEXT}, horizon {HORIZON}, {SAMPLES} samples; forecasts {cache_file.name}.",
        f"Fingerprint: `{fp}`. This was the one look: the result is locked.",
        "",
        "Data: " + "; ".join(notes),
    ]
    return "\n".join(lines) + "\n"


async def score(data_dir: Path, settings, instruments: dict, db, progress: Callable[[str], None] = print) -> tuple[Verdict, Path]:
    """The one look. Refuses if any forecast is missing or if Kronos was already scored."""
    from .round2 import holdout_record, record_holdout

    pl = plan()
    done = await holdout_record(db, STRATEGY, pl)
    if done:
        raise SystemExit(f"Kronos was already scored on {done.get('at')} ({done.get('verdict')}); the result is locked. "
                         f"Report: {done.get('report')}")
    data, notes = load_data(data_dir, instruments)
    path = cache_path(data_dir)
    cache = load_cache(path)
    daily = {s: daily_frame(data[s][0]) for s in SYMBOLS}
    missing = {s: n for s, n in missing_forecasts(daily, cache).items() if n}
    if missing:
        raise SystemExit(f"Forecasts missing ({missing}). Run `uv run --extra kronos tradeengine round4-kronos --forecast` "
                         "until it says nothing is left to do.")
    fp = fingerprint(settings, path)
    progress("Scoring Kronos and C2 over the same window (the one look) ...")
    kr, cr = run_both(data, settings, cache)
    k, c = _summary(kr, RULES), _summary(cr, RULES)
    ok, reasons = judge(k, c)
    v = Verdict(ok, reasons, k, c)
    out = data_dir / "reports" / f"round4-kronos-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render(v, fp, notes, path))
    (out / "result.json").write_text(json.dumps({"verdict": "PASS" if ok else "FAIL", "reasons": reasons, "kronos": k, "c2": c,
                                                 "fingerprint": fp, "config": config(), "forecasts": str(path)}, indent=2, default=str))
    trades = pd.DataFrame([asdict(t) for t in kr.trades])
    trades.to_csv(out / "kronos_trades.csv", index=False)
    await record_holdout(db, STRATEGY, {"at": datetime.now(UTC).isoformat(timespec="seconds"), "verdict": "PASS" if ok else "FAIL",
                                       "fingerprint": fp, "report": str(out / "report.md")}, pl)
    return v, out
