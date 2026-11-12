"""Pre-registered research rounds: round 2 (this file's constants) and round 3 (round3.py), both
run by the machinery here, driven by a Plan (coins, dates, rules, strategies, lock names).
Round 2 exactly as pre-registered on 12 Nov 2026 (plan: Claude Docs "Trading engine - Round 2
research plan (pre-registered)"). The constants ARE the pre-registration: changing them after
seeing a result defeats the point.

Development (`tradeengine round2 --dev`)
    Data is cut at DEV_END before anything is computed, so no holdout price can reach it.
    Per instrument and parameter set, a plain single-instrument backtest; walk-forward picks the
    parameters per window from trades that both entered AND exited inside the training window.
    Sit-out: if the best set's training profit factor is below 1.0 (or too few training trades),
    the instrument trades nothing in that test window.
    The chosen schedules are then replayed together as ONE portfolio through the live risk limits.
    Check 1: profit factor >= 1.2, worst-account drawdown < 10%, >= 200 trades.
    The basket for the holdout is frozen here: instruments with >= 30 development trades and PF >= 1.0.

Holdout (`tradeengine round2 --holdout <dev result.json>`)
    Only strategies that passed development, only their frozen basket, one run each, ever: the run is
    recorded with a fingerprint of the code and settings and a second run of the same strategy is
    refused. The walk-forward continues into the holdout with the same windows (training always on
    past data only). Check 2: profit factor >= 1.2, worst-account drawdown < 10%, >= 100 trades.
    result.json is in the format `tradeengine gate2-apply` reads.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from . import portfolio as pf_mod
from . import trend as trend_mod
from .backtest import DEFAULT_COSTS, Trade, cost_for, params_id
from .db import StateRow
from .history import load_h1
from .models import AssetKind
from .portfolio import (
    Leg,
    Limits,
    PortfolioMetrics,
    RateBook,
    portfolio_metrics,
    simulate,
    symbol_stats,
)
from .strategies import param_grid
from .trend import ROUND2_STRATEGIES, prep_key, prepare

# ------------------------------------------------------------------ the pre-registered constants
ROUND2_SYMBOLS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP", "EUR/JPY", "GBP/JPY",
    "XAU/USD", "BTC/USDT", "ETH/USDT",
]
DATA_FROM_FX = pd.Timestamp("2013-01-01", tz="UTC")
DATA_FROM_CRYPTO = pd.Timestamp("2018-01-01", tz="UTC")
DEV_END = pd.Timestamp("2023-11-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-01", tz="UTC")


@dataclass(frozen=True)
class Rules:
    dev_min_pf: float = 1.2
    dev_max_dd_pct: float = 10.0
    dev_min_trades: int = 200
    holdout_min_pf: float = 1.2
    holdout_max_dd_pct: float = 10.0
    holdout_min_trades: int = 100
    basket_min_trades: int = 30
    basket_min_pf: float = 1.0
    sitout_min_pf: float = 1.0
    # Statistical check (round 3): the 5th percentile of 10,000 seeded bootstrap means of trade R must be > 0.
    bootstrap: bool = False
    bootstrap_samples: int = 10_000
    bootstrap_pct: float = 5.0
    bootstrap_seed: int = 20261112


RULES = Rules()


@dataclass
class Plan:
    """One pre-registered round. Built by round2.plan() / round3.plan() from module constants at call time."""

    number: int
    symbols: list[str]
    data_from: dict  # "fx" (Forex and gold) / "crypto" -> first date used
    dev_end: pd.Timestamp
    holdout_end: pd.Timestamp
    rules: Rules
    strategies: dict
    wf_setting: str  # RuntimeSettings attribute with {strategy: [train months, test months]}

    @property
    def name(self) -> str:
        return f"round{self.number}"

    @property
    def lock_prefix(self) -> str:
        return f"{self.name}_holdout:"

    @property
    def counter_key(self) -> str:
        return f"{self.name}_dev_runs"

    def start_for(self, inst) -> pd.Timestamp:
        return self.data_from["crypto" if inst.kind is AssetKind.CRYPTO_SPOT else "fx"]

    def wf(self, settings) -> dict[str, list[int]]:
        return getattr(settings, self.wf_setting)


def plan() -> Plan:
    """Round 2, read from the module constants at call time."""
    return Plan(2, ROUND2_SYMBOLS, {"fx": DATA_FROM_FX, "crypto": DATA_FROM_CRYPTO}, DEV_END, HOLDOUT_END, RULES,
                ROUND2_STRATEGIES, "round2_walk_forward")


def bootstrap_lower_bound(rs: list[float], rules: Rules) -> float:
    """The `bootstrap_pct` percentile of `bootstrap_samples` resampled means of the trade results (seeded,
    so the same trades always give the same number). Above zero = about 95% confidence the average trade wins."""
    x = np.asarray(rs, dtype=float)
    if len(x) < 2:
        return float("-inf")
    rng = np.random.default_rng(rules.bootstrap_seed)
    means = x[rng.integers(0, len(x), size=(rules.bootstrap_samples, len(x)))].mean(axis=1)
    return float(np.percentile(means, rules.bootstrap_pct))
START_EQUITY = 10_000.0  # per venue; percentages are what matter
MAX_DATA_GAP = pd.Timedelta(days=4)  # longer than any weekend + holiday: a hole a download left


# ------------------------------------------------------------------ data
class IncompleteData(RuntimeError):
    """The pre-registered basket is not all there: a run on part of it would not be the planned test."""


def require_complete(notes: list[str], data: dict) -> None:
    missing = [n for n in notes if not any(n.startswith(f"{s}:") for s in data) or "WARNING" in n]
    if missing:
        raise IncompleteData("history missing or short for: " + "; ".join(missing)
                             + ". Run `./scripts/with-secrets.sh uv run tradeengine download-all --from 2013-01-01` first.")


def load_round2_data(data_dir: Path, instruments: dict, until: pd.Timestamp, pl: Plan | None = None) -> tuple[dict, list[str]]:
    """symbol -> (H1 frame cut to [data start, until), Instrument), plus one note per symbol."""
    pl = pl or plan()
    data, notes = {}, []
    for sym in pl.symbols:
        inst = instruments.get(sym)
        if inst is None:
            notes.append(f"{sym}: not in the instrument list; skipped")
            continue
        crypto = inst.kind is AssetKind.CRYPTO_SPOT
        df, _ = load_h1(data_dir, sym, crypto)
        if df is None or df.empty:
            notes.append(f"{sym}: no history file (run download-all --from 2013-01-01)")
            continue
        start = pl.start_for(inst)
        df = df[(df.index >= start) & (df.index < until)]
        if df.empty:
            notes.append(f"{sym}: no history between {start:%Y-%m-%d} and {until:%Y-%m-%d}")
            continue
        late = (df.index[0] - start).days > 31
        gaps = df.index.to_series().diff()
        worst = gaps.max() if len(gaps) > 1 else pd.Timedelta(0)
        holes = ""
        if worst > MAX_DATA_GAP:
            at = gaps.idxmax()
            holes = f" (WARNING: {worst.days}-day hole in the data before {at:%Y-%m-%d})"
        notes.append(f"{sym}: {len(df):,} hourly bars, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}"
                     + (f" (WARNING: starts {(df.index[0] - start).days} days after {start:%Y-%m-%d})" if late else "") + holes)
        data[sym] = (df, inst)
    return data, notes


def rate_book(data: dict) -> RateBook:
    return RateBook({sym: df["close"] for sym, (df, _) in data.items()})


# ------------------------------------------------------------------ walk-forward choice
@dataclass
class Window:
    train_start: str
    train_end: str
    test_end: str
    params_id: str | None  # None = sit out
    in_sample_trades: int
    in_sample_pf: float
    note: str = ""


def _pf(rs: list[float]) -> float:
    w = sum(r for r in rs if r > 0)
    lo = -sum(r for r in rs if r <= 0)
    return w / lo if lo > 0 else (10.0 if w > 0 else 0.0)


def choose_windows(trades_by_pid: dict[str, list[Trade]], first: pd.Timestamp, last: pd.Timestamp,
                   train_months: int, test_months: int, min_is_trades: int, sitout_min_pf: float) -> list[Window]:
    """Windows anchored at `first`, rolled by test_months while the training window fits the data."""
    parsed = {pid: [(pd.Timestamp(t.entry_time), pd.Timestamp(t.exit_time), t.r) for t in ts] for pid, ts in trades_by_pid.items()}
    out: list[Window] = []
    train_start = first
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        if train_end > last:
            break
        test_end = train_end + pd.DateOffset(months=test_months)
        best, best_score, best_n, best_pf = None, None, 0, 0.0
        for pid, ts in parsed.items():
            rs = [r for e, x, r in ts if train_start <= e and x < train_end]
            if len(rs) < min_is_trades:
                continue
            score = (min(_pf(rs), 10.0), sum(rs))
            if best_score is None or score > best_score:
                best, best_score, best_n, best_pf = pid, score, len(rs), _pf(rs)
        note = ""
        if best is None:
            note = f"sit out: no parameter set had {min_is_trades} completed training trades"
        elif best_pf < sitout_min_pf:
            note = f"sit out: best training profit factor {best_pf:.2f} < {sitout_min_pf}"
        chosen = best if not note else None
        out.append(Window(str(train_start), str(train_end), str(test_end), chosen, best_n, round(best_pf, 3), note))
        train_start = train_start + pd.DateOffset(months=test_months)
    return out


def schedule_for(windows: list[Window], start: pd.Timestamp, end: pd.Timestamp, grid_params: dict[str, dict]) -> list[tuple]:
    out = []
    for w in windows:
        a, b = max(pd.Timestamp(w.train_end), start), min(pd.Timestamp(w.test_end), end)
        if a < b:
            out.append((a, b, grid_params[w.params_id] if w.params_id else None))
    return out


# ------------------------------------------------------------------ one strategy, one period
@dataclass
class StrategyRun:
    strategy: str
    metrics: PortfolioMetrics
    per_symbol: dict[str, tuple[int, float, float]]
    windows: dict[str, list[Window]]
    rejections: dict[str, int]
    halted_at: str | None
    trades: list[Trade]
    equity: pd.DataFrame
    configurations: int
    period: tuple[str, str]
    notes: list[str] = field(default_factory=list)


def _legs_and_windows(strategy, data: dict, rates: RateBook, settings, progress, symbols: list[str] | None = None,
                      pl: Plan | None = None) -> tuple[dict, dict, dict, int]:
    """Standalone backtests of every parameter set; walk-forward windows per symbol."""
    pl = pl or plan()
    train_m, test_m = pl.wf(settings).get(strategy.name, [36, 6])
    grid = param_grid(strategy.grid)
    grid_params = {params_id(p): p for p in grid}
    risk_pct = float(settings.risk_per_trade_pct) / 100
    legs, windows = {}, {}
    for sym, (h1, inst) in data.items():
        if symbols is not None and sym not in symbols:
            continue
        progress(f"  {strategy.name:<20} {sym:<9} {len(grid)} parameter sets ...")
        crypto = inst.kind is AssetKind.CRYPTO_SPOT
        frames: dict[str, pd.DataFrame] = {}
        for p in grid:
            k = prep_key(strategy, p)
            if k not in frames:
                frames[k] = prepare(strategy, h1, p, crypto)
        some = next(iter(frames.values()))
        ready = some.index[some["ready"].to_numpy(dtype=bool)]
        if not len(ready):
            continue
        first, last = ready[0], some.index[-1] + pd.Timedelta(seconds=1)
        cost = cost_for(sym, settings.cost_model)
        by_pid = {}
        for p in grid:
            leg = Leg(sym, inst.kind, inst.venue, inst.base, inst.quote, cost, frames, [(first, last, p)],
                      prep_of=lambda q, s=strategy: prep_key(s, q))
            by_pid[params_id(p)] = simulate(strategy, [leg], rates, first, last, None, START_EQUITY, risk_pct).trades
        windows[sym] = choose_windows(by_pid, first, some.index[-1], train_m, test_m, strategy.min_is_trades, pl.rules.sitout_min_pf)
        legs[sym] = Leg(sym, inst.kind, inst.venue, inst.base, inst.quote, cost, frames, [],
                        prep_of=lambda q, s=strategy: prep_key(s, q))
    return legs, windows, grid_params, len(grid) * len(legs)


def run_strategy(strategy, data: dict, rates: RateBook, settings, start: pd.Timestamp | None, end: pd.Timestamp,
                 symbols: list[str] | None, progress=print, pl: Plan | None = None) -> StrategyRun:
    legs, windows, grid_params, configs = _legs_and_windows(strategy, data, rates, settings, progress, symbols, pl)
    use = [s for s in legs if symbols is None or s in symbols]
    if start is None:  # development: from the first test window anywhere
        firsts = [pd.Timestamp(w[0].train_end) for s, w in windows.items() if s in use and w]
        start = min(firsts) if firsts else end
    for s in use:
        legs[s].schedule = schedule_for(windows[s], start, end, grid_params)
    progress(f"  {strategy.name:<20} portfolio of {len(use)} instruments, {start:%Y-%m-%d} to {end:%Y-%m-%d} ...")
    res = simulate(strategy, [legs[s] for s in use], rates, start, end, Limits.from_settings(settings), START_EQUITY)
    return StrategyRun(
        strategy.name, portfolio_metrics(res), symbol_stats(res.trades), {s: windows[s] for s in use}, dict(res.rejections),
        res.halted_at, res.trades, res.equity, configs, (str(start), str(end)),
    )


# ------------------------------------------------------------------ verdicts
def _stat_reason(run: StrategyRun, r: Rules) -> tuple[float | None, list[str]]:
    if not r.bootstrap:
        return None, []
    lb = bootstrap_lower_bound([t.r for t in run.trades], r)
    if lb > 0:
        return lb, []
    return lb, [f"statistical check: {r.bootstrap_pct:g}th-percentile average R {lb:+.3f}, needs > 0"]


def judge_dev(run: StrategyRun, r: Rules | None = None) -> tuple[bool, list[str], list[str]]:
    m, r = run.metrics, r or RULES
    reasons = []
    if m.trades < r.dev_min_trades:
        reasons.append(f"{m.trades} trades, needs {r.dev_min_trades}")
    if m.profit_factor < r.dev_min_pf:
        reasons.append(f"profit factor {m.profit_factor:.2f}, needs {r.dev_min_pf}")
    if m.max_drawdown_pct >= r.dev_max_dd_pct:
        reasons.append(f"drawdown {m.max_drawdown_pct:.1f}%, limit {r.dev_max_dd_pct}%")
    reasons += _stat_reason(run, r)[1]
    basket = sorted(s for s, (n, p, _) in run.per_symbol.items() if n >= r.basket_min_trades and p >= r.basket_min_pf)
    return not reasons, reasons, basket


def judge_holdout(run: StrategyRun, basket: list[str], r: Rules | None = None) -> tuple[bool, list[str]]:
    m, r = run.metrics, r or RULES
    reasons = []
    if not basket:
        reasons.append("no instrument qualified on development data")
    if m.trades < r.holdout_min_trades:
        reasons.append(f"{m.trades} trades, needs {r.holdout_min_trades}")
    if m.profit_factor < r.holdout_min_pf:
        reasons.append(f"profit factor {m.profit_factor:.2f}, needs {r.holdout_min_pf}")
    if m.max_drawdown_pct >= r.holdout_max_dd_pct:
        reasons.append(f"drawdown {m.max_drawdown_pct:.1f}%, limit {r.holdout_max_dd_pct}%")
    reasons += _stat_reason(run, r)[1]
    return not reasons, reasons


def current_params(run: StrategyRun) -> dict[str, str]:
    """Parameters each instrument would trade with now (latest window); sit-outs are left out."""
    return {s: w[-1].params_id for s, w in run.windows.items() if w and w[-1].params_id}


# ------------------------------------------------------------------ fingerprint + one-look lock
def fingerprint(settings, pl: Plan | None = None) -> str:
    """Hash of the strategy, simulator and round code plus every setting that changes a result."""
    from . import round2 as me
    from . import round3 as r3

    pl = pl or plan()
    parts = [inspect.getsource(m) for m in (trend_mod, pf_mod, me, r3)]
    parts.append(json.dumps({k: asdict(cost_for(k, settings.cost_model)) for k in sorted(set(DEFAULT_COSTS) | set(pl.symbols))}, sort_keys=True))
    parts.append(json.dumps(asdict(Limits.from_settings(settings)), sort_keys=True))
    parts.append(json.dumps(pl.wf(settings), sort_keys=True))
    parts.append(json.dumps({"symbols": pl.symbols, "from": {k: str(v) for k, v in pl.data_from.items()}, "dev_end": str(pl.dev_end),
                             "holdout_end": str(pl.holdout_end), "rules": asdict(pl.rules), "strategies": sorted(pl.strategies)}, sort_keys=True))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


LOCK_PREFIX = "round2_holdout:"  # round 2's; each plan has its own (Plan.lock_prefix)


async def holdout_record(db, strategy: str, pl: Plan | None = None) -> dict | None:
    prefix = (pl or plan()).lock_prefix
    async with db.sessions() as s:
        row = await s.get(StateRow, prefix + strategy)
        return dict(row.value) if row else None


async def all_holdout_records(db, pl: Plan | None = None) -> dict[str, dict]:
    prefix = (pl or plan()).lock_prefix
    async with db.sessions() as s:
        rows = (await s.execute(select(StateRow).where(StateRow.key.like(prefix + "%")))).scalars().all()
    return {r.key[len(prefix):]: dict(r.value) for r in rows}


async def record_holdout(db, strategy: str, value: dict, pl: Plan | None = None) -> None:
    prefix = (pl or plan()).lock_prefix
    async with db.sessions() as s, s.begin():
        if await s.get(StateRow, prefix + strategy) is not None:
            raise RuntimeError(f"{strategy}: the holdout was already run")
        s.add(StateRow(key=prefix + strategy, value=value))


async def count_dev_run(db, pl: Plan | None = None) -> int:
    key = (pl or plan()).counter_key
    async with db.sessions() as s, s.begin():
        row = await s.get(StateRow, key)
        n = (row.value.get("n", 0) if row else 0) + 1
        if row is None:
            s.add(StateRow(key=key, value={"n": n}))
        else:
            row.value = {"n": n}
    return n


# ------------------------------------------------------------------ reports
def _fmt_pf(x: float) -> str:
    return "inf" if math.isinf(x) else f"{x:.2f}"


def _metrics_row(m: PortfolioMetrics) -> str:
    dd = ", ".join(f"{k} {v:.1f}%" for k, v in m.drawdown_by_venue.items())
    return (f"{m.trades} | {m.win_rate * 100:.0f}% | {_fmt_pf(m.profit_factor)} | {m.expectancy_r:+.2f} | "
            f"{m.max_drawdown_pct:.1f}% ({dd}) | {m.return_pct:+.1f}% | {m.sharpe:.2f}")


def render(kind: str, runs: list[StrategyRun], verdicts: dict[str, dict], notes: list[str], fp: str, extra: list[str],
           limits: Limits | None = None, pl: Plan | None = None) -> str:
    pl = pl or plan()
    r, lm = pl.rules, limits or Limits()
    title = "development (check 1)" if kind == "dev" else "HOLDOUT (check 2, one look)"
    out = [f"# Round {pl.number} — {title} ({datetime.now().strftime('%Y-%m-%d %H:%M')})", "",
           f"Code and settings fingerprint: `{fp}`", *extra, ""]
    stat = (f", and the statistical check ({r.bootstrap_pct:g}th percentile of {r.bootstrap_samples:,} bootstrap averages of "
            "trade R above zero)") if r.bootstrap else ""
    first = min(pl.data_from.values())
    if kind == "dev":
        out.append(f"Check 1 (development test windows, {first:%Y} – {pl.dev_end:%b %Y}): profit factor >= {r.dev_min_pf}, "
                   f"worst-account drawdown < {r.dev_max_dd_pct}%, >= {r.dev_min_trades} trades{stat}. Basket for the holdout: "
                   f">= {r.basket_min_trades} development trades and PF >= {r.basket_min_pf}.")
    else:
        out.append(f"Check 2 (holdout {pl.dev_end:%b %Y} – {pl.holdout_end - pd.Timedelta(days=1):%b %Y}, frozen basket): profit factor >= "
                   f"{r.holdout_min_pf}, worst-account drawdown < {r.holdout_max_dd_pct}%, >= {r.holdout_min_trades} trades{stat}.")
    out += [f"One portfolio per strategy with the live limits ({lm.risk_pct * 100:g}% risk, {lm.max_open} open, {lm.per_instrument} per "
            f"instrument, {lm.per_currency} per currency, {lm.leverage:g}:1 leverage, {lm.daily_loss_pct:g}% daily loss, "
            f"{lm.max_dd_pct:g}% drawdown halt). Costs: spread, slippage, commission, overnight funding (see cost_model). "
            "Profit factor on R multiples. Drawdown marked to market, per account (Capital.com, Binance); the worse one counts.", ""]
    out += ["## Data", *[f"- {n}" for n in notes], ""]
    lb_head, lb_rule = ("| Lower bound R ", "|---") if r.bootstrap else ("", "")
    out += ["## Verdict", "", f"| Strategy | Result | Trades | Win | PF | Exp R | Max DD (per account) | Return | Sharpe {lb_head}| Why not |",
            f"|---|---|---|---|---|---|---|---|---{lb_rule}|---|"]
    for run in runs:
        v = verdicts[run.strategy]
        lb = f" | {v['lower_bound']:+.3f}" if r.bootstrap and v.get("lower_bound") is not None else (" | —" if r.bootstrap else "")
        out.append(f"| {run.strategy} | {'PASS' if v['passed'] else 'not passed'} | {_metrics_row(run.metrics)}{lb} | {'; '.join(v['reasons']) or '—'} |")
    out.append("")
    for run in runs:
        v = verdicts[run.strategy]
        out += [f"## {run.strategy}", "", f"Period {run.period[0][:10]} to {run.period[1][:10]}; {run.configurations} configurations "
                f"tested (parameter sets x instruments)." + (f" Kill switch reached: {run.halted_at}." if run.halted_at else ""), ""]
        if kind == "dev":
            out.append(f"Frozen basket for the holdout: {', '.join(v['basket']) or 'none'}")
            out.append("")
        out += ["| Instrument | Trades | PF | Total R |", "|---|---|---|---|"]
        for sym in sorted(run.windows):
            n, p, tr = run.per_symbol.get(sym, (0, 0.0, 0.0))
            out.append(f"| {sym} | {n} | {_fmt_pf(p)} | {tr:+.1f} |")
        if run.rejections:
            out += ["", "Signals not taken (live limits): " + "; ".join(f"{k} {n}" for k, n in sorted(run.rejections.items(), key=lambda kv: -kv[1]))]
        out += ["", "Parameters per window (training PF, training trades); sit-outs marked:", ""]
        for sym, ws in sorted(run.windows.items()):
            out.append(f"- **{sym}** " + "; ".join(
                f"{w.train_end[:10]}: {w.params_id or 'SIT OUT'} ({w.in_sample_pf}, {w.in_sample_trades})" for w in ws))
        out.append("")
    out.append("A strategy that does not pass stays switched off. A pass is evidence on past data, not a promise; "
               "the demo forward test (Phase 4) is the final judge.")
    return "\n".join(out)


def save(dir_: Path, markdown: str, payload: dict, runs: list[StrategyRun]) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "report.md").write_text(markdown)
    (dir_ / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    pd.DataFrame([{**asdict(t)} for run in runs for t in run.trades]).to_csv(dir_ / "trades.csv", index=False)
    eq = {f"{run.strategy}:{c}": run.equity[c] for run in runs for c in run.equity.columns}
    if eq:
        pd.DataFrame(eq).to_csv(dir_ / "equity.csv")
    return dir_


def _run_payload(run: StrategyRun) -> dict:
    return {
        "metrics": asdict(run.metrics), "per_symbol": {s: {"trades": n, "pf": p, "total_r": tr} for s, (n, p, tr) in run.per_symbol.items()},
        "rejections": run.rejections, "halted_at": run.halted_at, "configurations": run.configurations, "period": run.period,
        "windows": {s: [asdict(w) for w in ws] for s, ws in run.windows.items()},
    }


# ------------------------------------------------------------------ drivers
def run_dev(data_dir: Path, settings, instruments: dict, strategies: dict | None = None, progress=print, pl: Plan | None = None):
    pl = pl or plan()
    strategies = strategies or pl.strategies
    data, notes = load_round2_data(data_dir, instruments, pl.dev_end, pl)
    require_complete(notes, data)
    rates = rate_book(data)
    runs, verdicts = [], {}
    for strat in strategies.values():
        run = run_strategy(strat, data, rates, settings, None, pl.dev_end, None, progress, pl)
        passed, reasons, basket = judge_dev(run, pl.rules)
        runs.append(run)
        verdicts[strat.name] = {"passed": passed, "reasons": reasons, "basket": basket, "lower_bound": _stat_reason(run, pl.rules)[0]}
    return runs, verdicts, notes


def run_holdout(data_dir: Path, settings, instruments: dict, dev: dict, strategies: dict | None = None, progress=print,
                pl: Plan | None = None):
    """Holdout for the strategies that passed development, on their frozen baskets."""
    pl = pl or plan()
    strategies = strategies or pl.strategies
    data, notes = load_round2_data(data_dir, instruments, pl.holdout_end, pl)
    require_complete(notes, data)
    rates = rate_book(data)
    runs, verdicts = [], {}
    for name, d in dev["strategies"].items():
        if not d["passed"] or name not in strategies:
            continue
        basket = d["basket"]
        run = run_strategy(strategies[name], data, rates, settings, pl.dev_end, pl.holdout_end, basket, progress, pl)
        passed, reasons = judge_holdout(run, basket, pl.rules)
        latest = current_params(run)
        runs.append(run)
        verdicts[name] = {"passed": passed, "reasons": reasons, "basket": basket, "lower_bound": _stat_reason(run, pl.rules)[0],
                          "enabled_symbols": [s for s in basket if s in latest] if passed else [], "latest_params": latest}
    return runs, verdicts, notes


def dev_payload(runs, verdicts, fp: str, run_no: int, pl: Plan | None = None) -> dict:
    pl = pl or plan()
    return {"kind": f"{pl.name}-dev", "fingerprint": fp, "run_at": datetime.now(UTC).isoformat(), "dev_run_number": run_no,
            "rules": asdict(pl.rules), "strategies": {r.strategy: {**verdicts[r.strategy], **_run_payload(r)} for r in runs}}


def holdout_payload(runs, verdicts, fp: str, dev_path: str, pl: Plan | None = None) -> dict:
    pl = pl or plan()
    return {
        "kind": f"{pl.name}-holdout", "fingerprint": fp, "run_at": datetime.now(UTC).isoformat(), "dev_result": dev_path,
        "rules": asdict(pl.rules),
        # gate2-apply reads "verdicts"
        "verdicts": [{"strategy": r.strategy, "passed": verdicts[r.strategy]["passed"], "reasons": verdicts[r.strategy]["reasons"],
                      "pooled": asdict(r.metrics), "enabled_symbols": verdicts[r.strategy]["enabled_symbols"],
                      "latest_params": verdicts[r.strategy]["latest_params"], "basket": verdicts[r.strategy]["basket"]} for r in runs],
        "strategies": {r.strategy: _run_payload(r) for r in runs},
    }
