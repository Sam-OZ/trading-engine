"""Event-driven backtester, walk-forward evaluation and the Gate 2 verdict.

Fill model (deliberately conservative):
- Bars are treated as MID prices. A buy pays half the spread above mid, a sell
  receives half below, so every round trip costs one full spread.
- A signal at the close of bar i enters at the OPEN of bar i+1, plus slippage.
- Stops: if a bar opens through the stop, the fill is the open (gap), else the
  stop level; both minus slippage. Targets fill at the target, no slippage.
- If one bar touches both stop and target, the STOP is assumed to fill first.
- Commission per side (crypto) and overnight funding per night held (CFDs) are charged.
- Position size: risk_pct of current equity lost if the stop fills at its level.

Walk-forward: parameters are chosen on a 12-month training window using only
trades that ENTERED and EXITED in that window, then judged on the next 3 months they have
never seen; the window rolls forward 3 months at a time. Gate 2 looks only at
those unseen (out-of-sample) trades.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .models import AssetKind, Side
from .strategies import (
    STRATEGIES,
    WARMUP_BARS,
    Signal,
    Strategy,
    columns,
    param_grid,
    trailed_stop,
)


# --------------------------------------------------------------------------------- costs
@dataclass(frozen=True)
class CostModel:
    spread: float = 0.0  # absolute, in price units (FX), full bid-ask
    spread_bps: float = 0.0  # relative, basis points of price (crypto)
    slippage: float = 0.0  # absolute, on market/stop fills
    slippage_bps: float = 0.0
    commission_rate: float = 0.0  # fraction of notional, per side
    funding_annual: float = 0.0  # fraction of notional per year, charged per night held (CFDs)

    def half_spread(self, price: float) -> float:
        return self.spread / 2 + price * self.spread_bps / 20000.0

    def slip(self, price: float) -> float:
        return self.slippage + price * self.slippage_bps / 10000.0


# Defaults are deliberately a little worse than Capital.com's advertised typical spreads
# and Binance's standard taker fee. Admin can override per symbol via settings.cost_model.
# Round 2 (12 Nov 2026): funding raised from 3% to 6% a year, long or short, because the
# round 2 strategies hold for weeks; these values may only ever go UP (pre-registered plan).
DEFAULT_COSTS: dict[str, CostModel] = {
    "EUR/USD": CostModel(spread=0.00010, slippage=0.00005, funding_annual=0.06),
    "GBP/USD": CostModel(spread=0.00015, slippage=0.00005, funding_annual=0.06),
    "USD/JPY": CostModel(spread=0.012, slippage=0.005, funding_annual=0.06),
    "AUD/USD": CostModel(spread=0.00012, slippage=0.00005, funding_annual=0.06),
    "USD/CAD": CostModel(spread=0.00023, slippage=0.00005, funding_annual=0.06),
    "USD/CHF": CostModel(spread=0.00018, slippage=0.00005, funding_annual=0.06),
    "NZD/USD": CostModel(spread=0.00020, slippage=0.00005, funding_annual=0.06),
    "EUR/GBP": CostModel(spread=0.00023, slippage=0.00005, funding_annual=0.06),
    "EUR/JPY": CostModel(spread=0.024, slippage=0.005, funding_annual=0.06),
    "GBP/JPY": CostModel(spread=0.036, slippage=0.008, funding_annual=0.06),
    "XAU/USD": CostModel(spread=0.60, slippage=0.10, funding_annual=0.06),
    "BTC/USDT": CostModel(spread_bps=1.0, slippage_bps=2.0, commission_rate=0.001),
    "ETH/USDT": CostModel(spread_bps=1.5, slippage_bps=2.0, commission_rate=0.001),
    "SOL/USDT": CostModel(spread_bps=3.0, slippage_bps=4.0, commission_rate=0.001),
}


def cost_for(symbol: str, overrides: dict | None = None) -> CostModel:
    base = DEFAULT_COSTS.get(symbol, CostModel(spread_bps=5.0, slippage_bps=5.0, commission_rate=0.001))
    if overrides and symbol in overrides:
        return CostModel(**{**asdict(base), **overrides[symbol]})
    return base


def quote_to_usd(quote: str, price: float) -> float:
    """Value of 1 unit of quote currency in USD, for the symbol's own price (JPY via USD/JPY)."""
    q = quote.upper()
    if q in ("USD", "USDT", "USDC"):
        return 1.0
    if q == "JPY":
        return 1.0 / price
    return 1.0  # other quotes are approximated; percentages and R-multiples are unaffected


# --------------------------------------------------------------------------------- single backtest
@dataclass
class Trade:
    symbol: str
    strategy: str
    params_id: str
    side: str
    entry_time: str
    exit_time: str
    entry: float
    exit: float
    stop0: float
    target: float | None
    units: float
    risk_amount: float
    pnl: float
    r: float
    costs: float
    bars: int
    reason: str


def params_id(p: dict) -> str:
    return ",".join(f"{k}={p[k]}" for k in sorted(p))


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    quote: str,
    strategy: Strategy,
    params: dict,
    cost: CostModel,
    allow_short: bool,
    risk_pct: float = 0.005,
    start_equity: float = 10_000.0,
) -> list[Trade]:
    cols = columns(df)
    times = df.index
    o, h, lo, c, atr = cols["open"], cols["high"], cols["low"], cols["close"], cols["atr"]
    n = len(df)
    equity = start_equity
    pid = params_id(params)
    trades: list[Trade] = []
    pos: dict | None = None
    pending: Signal | None = None

    def close_pos(i: int, price: float, reason: str) -> None:
        nonlocal pos, equity
        side = pos["side"]
        sign = 1 if side is Side.BUY else -1
        gross = (price - pos["entry"]) * pos["units"] * sign * quote_to_usd(quote, price)
        commission = cost.commission_rate * pos["units"] * price * quote_to_usd(quote, price)
        pnl = gross - commission - pos["entry_commission"] - pos["funding"]
        equity += pnl
        trades.append(
            Trade(
                symbol, strategy.name, pid, side.value, str(pos["time"]), str(times[i]), pos["entry"], price, pos["stop0"],
                pos["target"], pos["units"], pos["risk"], pnl, pnl / pos["risk"] if pos["risk"] else 0.0,
                commission + pos["entry_commission"] + pos["funding"], pos["bars"], reason,
            )
        )
        pos = None

    for i in range(min(WARMUP_BARS, n), n):
        if pending is not None and pos is None:
            sig, pending = pending, None
            half, slip = cost.half_spread(o[i]), cost.slip(o[i])
            entry = o[i] + half + slip if sig.side is Side.BUY else o[i] - half - slip
            valid = sig.stop < entry if sig.side is Side.BUY else sig.stop > entry
            if valid:
                per_unit = abs(entry - sig.stop) * quote_to_usd(quote, entry)
                risk = equity * risk_pct
                units = risk / per_unit if per_unit > 0 else 0.0
                if units > 0:
                    pos = {
                        "side": sig.side, "entry": entry, "stop": sig.stop, "stop0": sig.stop, "target": sig.target,
                        "units": units, "risk": risk, "time": times[i], "bars": 0, "best": entry,
                        "trail_atr": sig.trail_atr, "be_r": sig.breakeven_at_r, "max_bars": sig.max_bars,
                        "init_risk": abs(entry - sig.stop), "funding": 0.0, "day": times[i].normalize(),
                        "entry_commission": cost.commission_rate * units * entry * quote_to_usd(quote, entry), "entry_bar": i,
                    }

        if pos is not None:
            day = times[i].normalize()
            if cost.funding_annual and day > pos["day"]:
                nights = (day - pos["day"]).days
                pos["funding"] += pos["units"] * o[i] * quote_to_usd(quote, o[i]) * cost.funding_annual / 365.0 * nights
                pos["day"] = day
            half, slip = cost.half_spread(o[i]), cost.slip(o[i])
            first_bar = i == pos["entry_bar"]
            if pos["side"] is Side.BUY:
                if not first_bar and o[i] - half <= pos["stop"]:
                    close_pos(i, o[i] - half - slip, "stop (gap)")
                elif lo[i] - half <= pos["stop"]:
                    close_pos(i, pos["stop"] - slip, "stop")
                elif pos["target"] is not None and h[i] - half >= pos["target"]:
                    close_pos(i, pos["target"], "target")
            else:
                if not first_bar and o[i] + half >= pos["stop"]:
                    close_pos(i, o[i] + half + slip, "stop (gap)")
                elif h[i] + half >= pos["stop"]:
                    close_pos(i, pos["stop"] + slip, "stop")
                elif pos["target"] is not None and lo[i] + half <= pos["target"]:
                    close_pos(i, pos["target"], "target")
            if pos is not None:
                pos["bars"] += 1
                pos["best"] = max(pos["best"], c[i]) if pos["side"] is Side.BUY else min(pos["best"], c[i])
                pos["stop"] = trailed_stop(pos["side"], pos["entry"], pos["stop"], pos["init_risk"], pos["best"], atr[i], pos["trail_atr"], pos["be_r"])
                if pos["bars"] >= pos["max_bars"]:
                    exit_px = c[i] - cost.half_spread(c[i]) if pos["side"] is Side.BUY else c[i] + cost.half_spread(c[i])
                    close_pos(i, exit_px, "time")

        if pos is None and pending is None and i < n - 1:
            pending = strategy.signal_at(cols, i, params, allow_short)

    if pos is not None:
        exit_px = c[-1] - cost.half_spread(c[-1]) if pos["side"] is Side.BUY else c[-1] + cost.half_spread(c[-1])
        close_pos(n - 1, exit_px, "end of data")
    return trades


# --------------------------------------------------------------------------------- metrics
@dataclass
class Metrics:
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    avg_win_r: float
    avg_loss_r: float
    total_r: float
    max_drawdown_pct: float
    sharpe: float
    return_pct: float
    months: float


def metrics(trades: list[Trade], risk_pct: float = 0.005) -> Metrics:
    """Pools trades in exit order and compounds R x risk_pct, the way the account would."""
    if not trades:
        return Metrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    ts = sorted(trades, key=lambda t: t.exit_time)
    rs = np.array([t.r for t in ts])
    wins, losses = rs[rs > 0], rs[rs <= 0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    pf = float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    eq = np.cumprod(1 + rs * risk_pct)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    dd = float(((peak - eq) / peak).max()) * 100
    exits = pd.to_datetime([t.exit_time for t in ts], utc=True)
    daily = pd.Series(rs * risk_pct, index=exits).groupby(exits.normalize()).sum()
    start = pd.to_datetime(min(t.entry_time for t in ts), utc=True)
    days = pd.date_range(start.normalize(), exits.max().normalize(), freq="D")
    daily = daily.reindex(days, fill_value=0.0)
    sharpe = float(daily.mean() / daily.std() * math.sqrt(365)) if daily.std() > 0 else 0.0
    months = max((exits.max() - start).days / 30.44, 0.0)
    return Metrics(
        trades=len(ts),
        win_rate=float(len(wins) / len(rs)),
        profit_factor=pf,
        expectancy_r=float(rs.mean()),
        avg_win_r=float(wins.mean()) if len(wins) else 0.0,
        avg_loss_r=float(losses.mean()) if len(losses) else 0.0,
        total_r=float(rs.sum()),
        max_drawdown_pct=dd,
        sharpe=sharpe,
        return_pct=float((eq[-1] - 1) * 100),
        months=months,
    )


# --------------------------------------------------------------------------------- walk-forward
@dataclass
class WindowChoice:
    train_start: str
    train_end: str
    test_end: str
    params_id: str
    in_sample_trades: int
    in_sample_pf: float
    oos_trades: int


@dataclass
class SymbolResult:
    symbol: str
    strategy: str
    windows: list[WindowChoice] = field(default_factory=list)
    oos_trades: list[Trade] = field(default_factory=list)
    data_from: str = ""
    data_to: str = ""


def _pf(trades: list[Trade]) -> float:
    w = sum(t.r for t in trades if t.r > 0)
    lo = -sum(t.r for t in trades if t.r <= 0)
    return w / lo if lo > 0 else (10.0 if w > 0 else 0.0)


def walk_forward(
    h1: pd.DataFrame,
    symbol: str,
    quote: str,
    kind: AssetKind,
    strategy: Strategy,
    cost: CostModel,
    train_months: int = 12,
    test_months: int = 3,
    min_is_trades: int = 15,
    risk_pct: float = 0.005,
) -> SymbolResult:
    allow_short = kind is not AssetKind.CRYPTO_SPOT
    grid = param_grid(strategy.grid)
    frames: dict[str, pd.DataFrame] = {}
    all_trades: dict[str, list[Trade]] = {}
    for p in grid:
        key = json.dumps({k: p[k] for k in ("bb_std", "dc_n") if k in p}, sort_keys=True)
        if key not in frames:
            frames[key] = strategy.prepare(h1, p)
        all_trades[params_id(p)] = run_backtest(frames[key], symbol, quote, strategy, p, cost, allow_short, risk_pct)

    res = SymbolResult(symbol, strategy.name, data_from=str(h1.index[0]), data_to=str(h1.index[-1]))
    if len(h1) <= WARMUP_BARS:
        return res
    first = h1.index[WARMUP_BARS]
    last = h1.index[-1]
    train_start = first
    while True:
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last + pd.Timedelta(days=1):
            break
        best, best_score, best_is = None, -math.inf, []
        for pid, trades in all_trades.items():
            # Only trades that also EXITED inside the training window: a trade still open at train_end
            # would carry test-period prices into the parameter choice.
            is_t = [t for t in trades if train_start <= pd.Timestamp(t.entry_time) and pd.Timestamp(t.exit_time) < train_end]
            if len(is_t) < min_is_trades:
                continue
            score = (min(_pf(is_t), 10.0), sum(t.r for t in is_t))
            if best is None or score > best_score:
                best, best_score, best_is = pid, score, is_t
        if best is not None:
            oos = [t for t in all_trades[best] if train_end <= pd.Timestamp(t.entry_time) < test_end]
            res.oos_trades.extend(oos)
            res.windows.append(WindowChoice(str(train_start.date()), str(train_end.date()), str(test_end.date()), best, len(best_is), round(_pf(best_is), 3), len(oos)))
        else:
            res.windows.append(WindowChoice(str(train_start.date()), str(train_end.date()), str(test_end.date()), "(too few in-sample trades)", 0, 0.0, 0))
        train_start = train_start + pd.DateOffset(months=test_months)
    return res


# --------------------------------------------------------------------------------- Gate 2
@dataclass(frozen=True)
class GateRules:
    min_pf: float = 1.2
    max_dd_pct: float = 10.0
    min_trades: int = 200
    symbol_min_pf: float = 1.0
    symbol_min_trades: int = 30


@dataclass
class StrategyVerdict:
    strategy: str
    passed: bool
    reasons: list[str]
    pooled: Metrics
    per_symbol: dict[str, Metrics]
    enabled_symbols: list[str]
    latest_params: dict[str, str]


def judge(strategy: str, results: list[SymbolResult], rules: GateRules, risk_pct: float = 0.005) -> StrategyVerdict:
    per_symbol = {r.symbol: metrics(r.oos_trades, risk_pct) for r in results}
    eligible = [r for r in results if per_symbol[r.symbol].trades >= rules.symbol_min_trades and per_symbol[r.symbol].profit_factor >= rules.symbol_min_pf]
    pooled = metrics([t for r in eligible for t in r.oos_trades], risk_pct)
    reasons = []
    if pooled.trades < rules.min_trades:
        reasons.append(f"{pooled.trades} out-of-sample trades on eligible instruments, needs {rules.min_trades}")
    if pooled.profit_factor < rules.min_pf:
        reasons.append(f"profit factor {pooled.profit_factor:.2f}, needs {rules.min_pf}")
    if pooled.max_drawdown_pct >= rules.max_dd_pct:
        reasons.append(f"max drawdown {pooled.max_drawdown_pct:.1f}%, limit {rules.max_dd_pct}%")
    latest = {r.symbol: r.windows[-1].params_id for r in eligible if r.windows}
    return StrategyVerdict(strategy, not reasons, reasons, pooled, per_symbol, [r.symbol for r in eligible] if not reasons else [], latest)


# --------------------------------------------------------------------------------- report
def _fmt(m: Metrics) -> str:
    pf = "inf" if math.isinf(m.profit_factor) else f"{m.profit_factor:.2f}"
    return f"| {m.trades} | {m.win_rate * 100:.0f}% | {pf} | {m.expectancy_r:+.2f} | {m.max_drawdown_pct:.1f}% | {m.sharpe:.2f} | {m.return_pct:+.1f}% |"


def render_report(verdicts: list[StrategyVerdict], results: dict[str, list[SymbolResult]], rules: GateRules, data_notes: list[str]) -> str:
    out = [f"# Gate 2 — walk-forward backtest ({datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]
    out.append(f"Rules (out-of-sample only): profit factor >= {rules.min_pf}, max drawdown < {rules.max_dd_pct}%, "
               f">= {rules.min_trades} trades pooled; an instrument counts only with >= {rules.symbol_min_trades} trades and PF >= {rules.symbol_min_pf}.")
    out.append("Costs included: spread, slippage, commission (crypto), overnight funding (FX CFDs). Risk 0.5% per trade, compounding.")
    out += ["", "## Data", *[f"- {n}" for n in data_notes], ""]
    out += ["## Verdict", "", "| Strategy | Gate 2 | Trades | Win | PF | Exp R | Max DD | Sharpe | Return | Why not |", "|---|---|---|---|---|---|---|---|---|---|"]
    for v in verdicts:
        row = _fmt(v.pooled)[1:]
        out.append(f"| {v.strategy} | {'PASS' if v.passed else 'not passed'} |{row} {'; '.join(v.reasons) or '—'} |")
    out.append("")
    for v in verdicts:
        out += [f"## {v.strategy}", "", "| Instrument | Trades | Win | PF | Exp R | Max DD | Sharpe | Return |", "|---|---|---|---|---|---|---|---|"]
        for sym, m in v.per_symbol.items():
            out.append(f"| {sym} {_fmt(m)}")
        out += ["", "Parameters chosen per window (training PF, out-of-sample trades):", ""]
        for r in results[v.strategy]:
            out.append(f"- **{r.symbol}** " + "; ".join(f"{w.train_end}: {w.params_id} (PF {w.in_sample_pf}, {w.oos_trades} OOS)" for w in r.windows))
        out.append("")
    out.append("A strategy that does not pass stays switched off. Passing here is evidence on past data, not a promise; Phase 4 forward-tests on demo before anything else changes.")
    return "\n".join(out)


def save_report(dir_: Path, verdicts: list[StrategyVerdict], results: dict[str, list[SymbolResult]], markdown: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "report.md").write_text(markdown)
    payload = {
        "verdicts": [
            {
                "strategy": v.strategy, "passed": v.passed, "reasons": v.reasons, "pooled": asdict(v.pooled),
                "per_symbol": {k: asdict(m) for k, m in v.per_symbol.items()}, "enabled_symbols": v.enabled_symbols,
                "latest_params": v.latest_params,
            }
            for v in verdicts
        ],
        "windows": {s: [{"symbol": r.symbol, "windows": [asdict(w) for w in r.windows]} for r in rs] for s, rs in results.items()},
    }
    (dir_ / "result.json").write_text(json.dumps(payload, indent=2, default=str))
    rows = [asdict(t) for rs in results.values() for r in rs for t in r.oos_trades]
    pd.DataFrame(rows).to_csv(dir_ / "oos_trades.csv", index=False)
    return dir_


__all__ = [
    "STRATEGIES", "CostModel", "DEFAULT_COSTS", "GateRules", "Trade", "cost_for", "judge", "metrics", "render_report",
    "run_backtest", "save_report", "walk_forward",
]


# --------------------------------------------------------------------------------- Gate 2 driver
def run_gate2(data_dir: Path, settings, instruments: dict, strategies: dict | None = None, progress=print):
    """Walk-forward every strategy on every allowed instrument's H1 history and judge Gate 2."""
    from .history import load_h1

    strategies = strategies or STRATEGIES
    rules = GateRules(
        float(settings.gate2_min_pf), float(settings.gate2_max_dd_pct), settings.gate2_min_trades,
        float(settings.gate2_symbol_min_pf), settings.gate2_symbol_min_trades,
    )
    risk_pct = float(settings.risk_per_trade_pct) / 100
    data, notes = {}, []
    for sym in settings.instruments:
        inst = instruments.get(sym)
        if inst is None:
            continue
        df, note = load_h1(data_dir, sym, inst.kind is AssetKind.CRYPTO_SPOT)
        notes.append(note)
        if df is not None and len(df) > WARMUP_BARS + 24 * 30 * (settings.walk_forward_train_months + settings.walk_forward_test_months):
            data[sym] = (df, inst)
        elif df is not None:
            notes.append(f"{sym}: too little history for {settings.walk_forward_train_months}+{settings.walk_forward_test_months} months walk-forward; skipped")
    results: dict[str, list[SymbolResult]] = {}
    verdicts = []
    for name, strat in strategies.items():
        results[name] = []
        for sym, (df, inst) in data.items():
            progress(f"  {name:<15} {sym:<9} ...")
            results[name].append(
                walk_forward(df, sym, inst.quote, inst.kind, strat, cost_for(sym, settings.cost_model),
                             settings.walk_forward_train_months, settings.walk_forward_test_months, risk_pct=risk_pct)
            )
        verdicts.append(judge(name, results[name], rules, risk_pct))
    return verdicts, results, rules, notes


def parse_params_id(pid: str) -> dict:
    out = {}
    for part in pid.split(","):
        k, _, v = part.partition("=")
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out
