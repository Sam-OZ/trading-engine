"""Portfolio backtest: every instrument replayed in time order through the live risk checks.

Round 1 tested each instrument alone and added the trades together, which ignores that live
trading can hold only so many positions at once, and measured drawdown from closed trades only.
This simulator applies the same limits as risk.evaluate() for strategy trades:

- kill switch: a venue reaching the maximum drawdown halts ALL new entries (live, the halt
  engages the engine-wide kill switch); open positions keep their stops
- daily loss limit per venue (UTC day): no new entries for the rest of that day
- max open positions (engine-wide), 1 per instrument, max per currency (Forex and gold)
- leverage cap per CFD venue (size shrinks to fit, as live), free balance for spot crypto
- risk per trade = risk_pct of the venue's current equity (open positions marked to market)

Each venue is its own account (Capital.com for CFDs, Binance for spot crypto), as live; both start
at the same notional amount, so percentages are what matter. Equity is marked to market at every
bar close (exit side of the spread), and drawdown is measured on that.

Fill model is the round 1 one: bars are mid prices, a buy pays half the spread above, entries at the
next bar's open plus slippage, stops fill at the stop (or the open if it gapped through) minus
slippage, a bar touching stop and target fills the stop, commission and overnight funding charged.
A signal exit (strategy.exit_at) fills at the next bar's open.

With limits=None and one instrument the same loop is a plain single-instrument backtest; the
walk-forward parameter choice uses that.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import CostModel, Trade, params_id
from .models import AssetKind, Side, Venue
from .strategies import columns, trailed_stop

# Quote currency -> (rate series, invert): USD value of one unit of the currency.
_TO_USD = {
    "EUR": ("EUR/USD", False), "GBP": ("GBP/USD", False), "AUD": ("AUD/USD", False), "NZD": ("NZD/USD", False),
    "JPY": ("USD/JPY", True), "CHF": ("USD/CHF", True), "CAD": ("USD/CAD", True), "XAU": ("XAU/USD", False),
}
_USD = {"USD", "USDT", "USDC", "FDUSD"}


class RateBook:
    """USD conversion from the hourly closes of the instruments themselves (no look-ahead: a close
    is used only once its hour has ended)."""

    def __init__(self, series: dict[str, pd.Series]):
        self._t: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        for sym, s in series.items():
            s = s.dropna().sort_index()
            self._t[sym] = (s.index + pd.Timedelta(hours=1)).as_unit("ns").asi8  # available from the bar's close
            self._v[sym] = s.to_numpy(dtype=float)

    def _asof(self, sym: str, t: pd.Timestamp) -> float | None:
        ts = self._t.get(sym)
        if ts is None or not len(ts):
            return None
        k = int(np.searchsorted(ts, pd.Timestamp(t).as_unit("ns").value, side="right")) - 1
        if k < 0:
            k = 0  # before the first close: use the first known rate rather than none
        return float(self._v[sym][k])

    def to_usd(self, ccy: str, t: pd.Timestamp, own: tuple[str, float] | None = None) -> float:
        c = ccy.upper()
        if c in _USD:
            return 1.0
        route = _TO_USD.get(c)
        if route is not None:
            sym, inv = route
            px = own[1] if own and own[0] == sym else self._asof(sym, t)
            if px:
                return 1.0 / px if inv else px
        return 1.0  # unknown currency: percentages and R-multiples are unaffected


@dataclass(frozen=True)
class Limits:
    risk_pct: float = 0.005
    daily_loss_pct: float = 2.0
    max_dd_pct: float = 10.0
    max_open: int = 4
    per_instrument: int = 1
    per_currency: int = 2
    leverage: float = 5.0

    @classmethod
    def from_settings(cls, s) -> Limits:
        return cls(float(s.risk_per_trade_pct) / 100, float(s.daily_loss_limit_pct), float(s.max_drawdown_pct),
                   s.max_open_positions, s.max_positions_per_instrument, s.max_positions_per_currency, float(s.max_effective_leverage))


@dataclass
class Leg:
    """One instrument in a simulation: prepared frames per preparation key, and which parameters
    apply when ([start, end) windows; params None = sit out, no new entries)."""

    symbol: str
    kind: AssetKind
    venue: Venue
    base: str
    quote: str
    cost: CostModel
    frames: dict[str, pd.DataFrame]
    schedule: list[tuple[pd.Timestamp, pd.Timestamp, dict | None]]
    prep_of: object = None  # callable(params) -> key into frames
    # Perpetual futures (round 5): the exchange's real funding rate at each funding time (index UTC, value as a
    # fraction of notional). Longs pay a positive rate, shorts receive it; a negative rate the other way round.
    funding: pd.Series | None = None

    @property
    def allow_short(self) -> bool:
        return self.kind is not AssetKind.CRYPTO_SPOT

    def params_at(self, t: pd.Timestamp) -> dict | None:
        for start, end, p in self.schedule:
            if start <= t < end:
                return p
        return None


@dataclass
class SimResult:
    trades: list[Trade]
    equity: pd.DataFrame  # index time, one column per venue (marked to market)
    rejections: Counter = field(default_factory=Counter)
    halted_at: str | None = None


@dataclass
class _Acct:
    cash: float
    peak: float
    eq: float
    sod: float
    sod_day: object = None
    halted_day: object = None  # daily-loss block


@dataclass
class _Pos:
    side: Side
    entry: float
    stop: float
    stop0: float
    target: float | None
    units: float
    risk: float
    time: pd.Timestamp
    entry_i: int
    trail_atr: float | None
    be_r: float | None
    max_bars: int | None
    activate_r: float
    init_risk: float
    params: dict
    pid: str
    reason: str
    notional: float
    entry_commission: float
    day: pd.Timestamp
    bars: int = 0
    best: float = 0.0
    funding: float = 0.0
    fund_mark: int = 0  # ns: funding payments after this instant are still to be applied


def simulate(
    strategy,
    legs: list[Leg],
    rates: RateBook,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    limits: Limits | None,
    start_equity: float = 10_000.0,
    risk_pct: float | None = None,
) -> SimResult:
    """Event-driven replay over [period_start, period_end). Entries only from signals whose bar
    lies in the period; anything still open at the end is closed at the last close."""
    risk_pct = limits.risk_pct if limits else (risk_pct or 0.005)
    cols_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    index: dict[str, pd.DatetimeIndex] = {}
    for leg in legs:
        idx = None
        for k, f in leg.frames.items():
            if idx is None:
                idx = f.index
            elif not f.index.equals(idx):
                raise ValueError(f"{leg.symbol}: prepared frames disagree on bar times ({k})")
            cols_cache[(leg.symbol, k)] = columns(f)
        index[leg.symbol] = idx if idx is not None else pd.DatetimeIndex([], tz="UTC")
    fund_arr: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for leg in legs:
        if leg.funding is not None:
            f = leg.funding.dropna().sort_index()
            fund_arr[leg.symbol] = (f.index.as_unit("ns").asi8, f.to_numpy(dtype=float))

    def key_of(leg: Leg, p: dict) -> str:
        return leg.prep_of(p)

    # Unified timeline of bar times in the period.
    events: dict[pd.Timestamp, list[tuple[Leg, int]]] = {}
    for leg in sorted(legs, key=lambda x: x.symbol):
        idx = index[leg.symbol]
        lo, hi = idx.searchsorted(period_start), idx.searchsorted(period_end)
        for i in range(lo, hi):
            events.setdefault(idx[i], []).append((leg, i))
    times = sorted(events)

    venues = sorted({leg.venue for leg in legs}, key=lambda v: v.value)
    acct = {v: _Acct(start_equity, start_equity, start_equity, start_equity) for v in venues}
    pos: dict[str, _Pos | None] = {leg.symbol: None for leg in legs}
    pending: dict[str, tuple | None] = {leg.symbol: None for leg in legs}
    pending_exit: dict[str, bool] = {leg.symbol: False for leg in legs}
    last_close: dict[str, float] = {}
    legs_by = {leg.symbol: leg for leg in legs}
    trades: list[Trade] = []
    rej: Counter = Counter()
    curve_t: list[pd.Timestamp] = []
    curve_v: list[list[float]] = []
    halted = False
    halted_at: str | None = None

    def q2u(leg: Leg, t: pd.Timestamp, px: float) -> float:
        return rates.to_usd(leg.quote, t, own=(leg.symbol, px))

    def close(leg: Leg, t: pd.Timestamp, price: float, reason: str) -> None:
        p = pos[leg.symbol]
        sign = p.side.sign
        conv = q2u(leg, t, price)
        gross = (price - p.entry) * p.units * sign * conv
        commission = leg.cost.commission_rate * p.units * price * conv
        pnl = gross - commission - p.entry_commission - p.funding
        acct[leg.venue].cash += pnl
        trades.append(Trade(leg.symbol, strategy.name, p.pid, p.side.value, str(p.time), str(t), p.entry, price, p.stop0, p.target,
                            p.units, p.risk, pnl, pnl / p.risk if p.risk else 0.0, commission + p.entry_commission + p.funding,
                            p.bars, reason, funding=p.funding))
        pos[leg.symbol] = None

    def open_positions() -> list[tuple[Leg, _Pos]]:
        return [(legs_by[s], p) for s, p in pos.items() if p is not None]

    def try_enter(leg: Leg, sig, params: dict, i: int, t: pd.Timestamp, o: float) -> None:
        a = acct[leg.venue]
        if limits is not None:
            if halted:
                rej["kill switch (drawdown halt)"] += 1
                return
            if a.halted_day == t.date():
                rej["daily loss limit"] += 1
                return
            opens = open_positions()
            if len(opens) >= limits.max_open:
                rej["max open positions"] += 1
                return
            if sum(1 for lg, _ in opens if lg.symbol == leg.symbol) >= limits.per_instrument:
                rej["already holding instrument"] += 1
                return
            if leg.kind in (AssetKind.FX, AssetKind.METAL):
                for ccy in (leg.base, leg.quote):
                    n = sum(1 for lg, _ in opens if lg.kind in (AssetKind.FX, AssetKind.METAL) and ccy in (lg.base, lg.quote))
                    if n >= limits.per_currency:
                        rej[f"max positions per currency ({ccy})"] += 1
                        return
        half, slip = leg.cost.half_spread(o), leg.cost.slip(o)
        entry = o + half + slip if sig.side is Side.BUY else o - half - slip
        if not ((sig.stop < entry) if sig.side is Side.BUY else (sig.stop > entry)):
            rej["stop on the wrong side after the gap"] += 1
            return
        conv = q2u(leg, t, entry)
        per_unit = abs(entry - sig.stop) * conv
        if per_unit <= 0:
            return
        units = a.eq * risk_pct / per_unit
        unit_notional = entry * conv
        if limits is not None:
            if leg.kind is not AssetKind.CRYPTO_SPOT:
                used = sum(p.notional for lg, p in open_positions() if lg.venue is leg.venue)
                room = a.eq * limits.leverage - used
                if units * unit_notional > room:
                    units = room / unit_notional if room > 0 else 0.0
                    if units <= 0:
                        rej["leverage cap"] += 1
                        return
                    rej["size reduced by leverage cap"] += 1
            else:
                committed = sum(p.notional for lg, p in open_positions() if lg.venue is leg.venue)
                free = a.cash - committed
                if units * unit_notional > free:
                    units = free * 0.98 / unit_notional if free > 0 else 0.0
                    if units <= 0:
                        rej["no free balance"] += 1
                        return
                    rej["size reduced to free balance"] += 1
        risk = units * per_unit
        pos[leg.symbol] = _Pos(
            sig.side, entry, sig.stop, sig.stop, sig.target, units, risk, t, i, sig.trail_atr, sig.breakeven_at_r, sig.max_bars,
            sig.trail_activate_r, abs(entry - sig.stop), params, params_id(params), sig.reason, units * unit_notional,
            leg.cost.commission_rate * units * entry * conv, t.normalize(), best=entry, fund_mark=pd.Timestamp(t).value,
        )

    def mark(t: pd.Timestamp) -> None:
        nonlocal halted, halted_at
        for v in venues:
            a = acct[v]
            unreal = 0.0
            for lg, p in open_positions():
                if lg.venue is not v:
                    continue
                c = last_close.get(lg.symbol, p.entry)
                half = lg.cost.half_spread(c)
                exit_px = c - half if p.side is Side.BUY else c + half
                unreal += (exit_px - p.entry) * p.units * p.side.sign * q2u(lg, t, exit_px) - p.entry_commission - p.funding
            a.eq = a.cash + unreal
            a.peak = max(a.peak, a.eq)
            if limits is not None:
                if a.peak > 0 and (a.peak - a.eq) / a.peak * 100 >= limits.max_dd_pct and not halted:
                    halted, halted_at = True, f"{v.value} drawdown reached {limits.max_dd_pct}% at {t}"
                if a.sod > 0 and (a.sod - a.eq) / a.sod * 100 >= limits.daily_loss_pct:
                    a.halted_day = t.date()
        curve_t.append(t)
        curve_v.append([acct[v].eq for v in venues])

    for t in times:
        day = t.date()
        for v in venues:  # start-of-day equity per venue (UTC day), as live
            if acct[v].sod_day != day:
                acct[v].sod_day, acct[v].sod = day, acct[v].eq
        for leg, i in events[t]:
            sym = leg.symbol
            params_now = leg.params_at(t)
            cols_now = cols_cache[(sym, key_of(leg, params_now))] if params_now is not None else None
            some = cols_cache[(sym, next(iter(leg.frames)))]
            o, h, lo, c = some["open"][i], some["high"][i], some["low"][i], some["close"][i]
            p = pos[sym]
            # overnight funding
            if p is not None and leg.cost.funding_annual:
                d = t.normalize()
                if d > p.day:
                    p.funding += p.units * o * q2u(leg, t, o) * leg.cost.funding_annual / 365.0 * (d - p.day).days
                    p.day = d
            # perpetual-futures funding paid or received since the last bar (a payment at the entry instant is not
            # held; one at an exit at this bar's open is: the live engine trades after the 00:00 UTC funding)
            if p is not None and sym in fund_arr:
                ft, fv = fund_arr[sym]
                t_ns = pd.Timestamp(t).value
                a, b = np.searchsorted(ft, p.fund_mark, side="right"), np.searchsorted(ft, t_ns, side="right")
                if b > a:
                    p.funding += p.side.sign * float(fv[a:b].sum()) * p.units * o * q2u(leg, t, o)
                p.fund_mark = max(p.fund_mark, t_ns)
            # signal exit decided at the previous close
            if p is not None and pending_exit[sym]:
                half, slip = leg.cost.half_spread(o), leg.cost.slip(o)
                close(leg, t, o - half - slip if p.side is Side.BUY else o + half + slip, "signal exit")
                p = None
            pending_exit[sym] = False
            # entry decided at the previous close
            if pending[sym] is not None and p is None:
                sig, sp = pending[sym]
                try_enter(leg, sig, sp, i, t, o)
                p = pos[sym]
            pending[sym] = None
            # stop / target on this bar
            if p is not None:
                half, slip = leg.cost.half_spread(o), leg.cost.slip(o)
                first = i == p.entry_i
                if p.side is Side.BUY:
                    if not first and o - half <= p.stop:
                        close(leg, t, o - half - slip, "stop (gap)")
                    elif lo - half <= p.stop:
                        close(leg, t, p.stop - slip, "stop")
                    elif p.target is not None and h - half >= p.target:
                        close(leg, t, p.target, "target")
                else:
                    if not first and o + half >= p.stop:
                        close(leg, t, o + half + slip, "stop (gap)")
                    elif h + half >= p.stop:
                        close(leg, t, p.stop + slip, "stop")
                    elif p.target is not None and lo + half <= p.target:
                        close(leg, t, p.target, "target")
                p = pos[sym]
            last_close[sym] = c
            # manage at the close
            if p is not None:
                pc = cols_cache[(sym, key_of(leg, p.params))]
                p.bars += 1
                p.best = max(p.best, c) if p.side is Side.BUY else min(p.best, c)
                p.stop = trailed_stop(p.side, p.entry, p.stop, p.init_risk, p.best, pc["atr"][i], p.trail_atr, p.be_r, p.activate_r)
                if p.max_bars is not None and p.bars >= p.max_bars:
                    half = leg.cost.half_spread(c)
                    close(leg, t, c - half if p.side is Side.BUY else c + half, "time")
                    p = None
                elif hasattr(strategy, "exit_at") and strategy.exit_at(pc, i, p.side, p.params):
                    pending_exit[sym] = True
            # new signal at the close (also on the bar a signal exit was decided: a reversal)
            last_bar = i >= len(index[sym]) - 1
            if cols_now is not None and not last_bar and (p is None or pending_exit[sym]):
                sig = strategy.signal_at(cols_now, i, params_now, leg.allow_short)
                if sig is not None and (p is None or sig.side is not p.side):
                    pending[sym] = (sig, params_now)
        mark(t)

    # close what is still open at the end of the period
    for leg in legs:
        p = pos[leg.symbol]
        if p is not None:
            c = last_close.get(leg.symbol, p.entry)
            half = leg.cost.half_spread(c)
            close(leg, times[-1], c - half if p.side is Side.BUY else c + half, "end of period")
    if times:
        mark(times[-1])
    equity = pd.DataFrame(curve_v, index=pd.DatetimeIndex(curve_t, name="time"), columns=[v.value for v in venues])
    equity = equity[~equity.index.duplicated(keep="last")]
    return SimResult(trades, equity, rej, halted_at)


# --------------------------------------------------------------------------------- metrics
@dataclass
class PortfolioMetrics:
    trades: int
    win_rate: float
    profit_factor: float
    expectancy_r: float
    total_r: float
    max_drawdown_pct: float  # the worse of the venues (each venue halts on its own)
    drawdown_by_venue: dict[str, float]
    return_pct: float  # combined equity
    sharpe: float  # daily combined equity returns, annualised (sqrt 365)
    months: float


def _pf(rs: np.ndarray) -> float:
    w, lo = rs[rs > 0].sum(), -rs[rs <= 0].sum()
    return float(w / lo) if lo > 0 else (math.inf if w > 0 else 0.0)


def portfolio_metrics(res: SimResult) -> PortfolioMetrics:
    rs = np.array([t.r for t in res.trades], dtype=float)
    dd = {}
    for v in res.equity.columns:
        eq = res.equity[v].to_numpy(dtype=float)
        if len(eq):
            peak = np.maximum.accumulate(eq)
            dd[v] = float(((peak - eq) / peak).max() * 100)
    total = res.equity.sum(axis=1) if len(res.equity.columns) else pd.Series(dtype=float)
    ret = float((total.iloc[-1] / total.iloc[0] - 1) * 100) if len(total) > 1 else 0.0
    sharpe = 0.0
    if len(total) > 2:
        daily = total.groupby(total.index.normalize()).last().pct_change().dropna()
        if daily.std() > 0:
            sharpe = float(daily.mean() / daily.std() * math.sqrt(365))
    months = float((res.equity.index[-1] - res.equity.index[0]).days / 30.44) if len(res.equity) > 1 else 0.0
    if not len(rs):
        return PortfolioMetrics(0, 0.0, 0.0, 0.0, 0.0, max(dd.values(), default=0.0), dd, ret, sharpe, months)
    return PortfolioMetrics(
        trades=len(rs), win_rate=float((rs > 0).mean()), profit_factor=_pf(rs), expectancy_r=float(rs.mean()),
        total_r=float(rs.sum()), max_drawdown_pct=max(dd.values(), default=0.0), drawdown_by_venue=dd,
        return_pct=ret, sharpe=sharpe, months=months,
    )


def symbol_stats(trades: list[Trade]) -> dict[str, tuple[int, float, float]]:
    """symbol -> (trades, profit factor, total R)."""
    out: dict[str, list[float]] = {}
    for t in trades:
        out.setdefault(t.symbol, []).append(t.r)
    return {s: (len(r), _pf(np.array(r)), float(sum(r))) for s, r in out.items()}
