"""Live strategy runner: once per hour, the same strategy code the backtesters used.

At each hour close (plus a short grace period so the broker has the bar):
1. For every instrument with an enabled strategy (or an open strategy position), the latest
   completed H1 bars. Forex and gold start from the local Dukascopy file (Capital.com's demo API
   keeps only ~12 days of hourly prices) with Capital.com's recent bars appended; a gap between the
   two is filled from Dukascopy, and a symbol with a gap that cannot be filled is not traded.
2. prepare() turns the hourly bars into the strategy's own bars (hourly, 4-hour or daily, completed
   bars only); signal_at(last bar) -> a TradeIntent through the normal risk + execution path. A
   signal is acted on only in the first hour of trading after its bar closed (as the backtest enters
   at the next bar's open); later passes ignore it. The signal time is the bar's open time, so the
   client order id is the same however many times an hour is processed: a restart cannot double an order.
3. Manage open strategy positions exactly as the backtesters do: breakeven, ATR trail (on the
   strategy's own bars), time exit, and signal exits (exit_at). Stops only ever tighten, and the
   new stop is placed at the broker.

Nothing runs until `strategies_enabled` lists a strategy for a symbol (set by
`tradeengine gate2-apply`). An empty setting means the runner is idle.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pandas as pd
from sqlalchemy import select

from .db import PositionMetaRow, PositionRow
from .execution import journal_to_broker
from .history import (
    bars_to_frame,
    download_dukascopy_h1,
    history_path,
    load_h1,
    write_bars,
)
from .ids import client_order_id
from .models import (
    AssetKind,
    OrderStatus,
    PositionStatus,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from .sizing import round_to_tick
from .state import log_event
from .strategies import STRATEGIES, WARMUP_BARS, columns, trailed_stop
from .trend import ROUND2_STRATEGIES, ROUND3_STRATEGIES, prep_key, prepare

log = logging.getLogger(__name__)

ALL_STRATEGIES = {**STRATEGIES, **ROUND2_STRATEGIES, **ROUND3_STRATEGIES}

# Daily EMA200s need a long run-in before they stop depending on where the data starts:
# 1,100 days leaves ~600 daily bars after the seed, so live values match the backtest's.
HISTORY_DAYS = 1100
REFRESH_DAYS = 3  # after the first load, only the last few days are fetched each hour
SEED_RECENT_DAYS = 14  # Capital.com recent bars appended to the local file at the first load
MAX_GAP = pd.Timedelta(days=4)  # longer than any weekend + holiday: a real hole in the data
MAX_DATA_AGE = pd.Timedelta(hours=2)  # act on a signal only while the hourly data is current
GRACE_SECONDS = 20


def _dec(x: float | None) -> Decimal | None:
    return None if x is None else Decimal(repr(float(x)))


def _px(x: float | None, tick: Decimal | None) -> Decimal | None:
    """A price as the broker will hold it: rounded to the instrument's tick."""
    d = _dec(x)
    return d if d is None or not tick else round_to_tick(d, tick)


def _closes_at(frame: pd.DataFrame, i: int) -> pd.Timestamp:
    if "closes_at" in frame.columns:
        return pd.Timestamp(frame["closes_at"].iloc[i])
    return frame.index[i] + pd.Timedelta(hours=1)


def _fresh(frame: pd.DataFrame, i: int, h1: pd.DataFrame, now: datetime) -> bool:
    """True in the first hour of trading after bar i closed (at most one hourly bar since), and only
    while the hourly data is live (its last bar closed within MAX_DATA_AGE). After a weekend the first
    pass that qualifies is the one after Sunday's first hourly bar has closed."""
    live = pd.Timestamp(now) - (h1.index[-1] + pd.Timedelta(hours=1)) <= MAX_DATA_AGE
    return live and int((h1.index >= _closes_at(frame, i)).sum()) <= 1


class StrategyRunner:
    def __init__(self, engine, strategies: dict | None = None):
        self.e = engine
        self.strategies = strategies or ALL_STRATEGIES
        self._cache: dict[str, pd.DataFrame] = {}
        self._frames: dict[tuple, pd.DataFrame] = {}  # per on_hour pass

    async def run_forever(self) -> None:
        while True:
            now = datetime.now(UTC)
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            await asyncio.sleep((next_hour - now).total_seconds() + GRACE_SECONDS)
            try:
                await self.on_hour(datetime.now(UTC))
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                log.exception("strategy runner failed")
                await log_event(self.e.db, "ERROR", "runner", f"Hourly run failed: {ex}")

    async def _seed_from_file(self, inst, now: datetime) -> pd.DataFrame:
        """Forex/gold: local hourly history + Capital.com's recent bars, with any gap filled from Dukascopy."""
        adapter = self.e.adapters[inst.venue]
        data_dir = self.e.cfg.data_dir
        local, _ = load_h1(data_dir, inst.symbol, crypto=False)
        bars = await adapter.history(inst, "H1", now - timedelta(days=SEED_RECENT_DAYS), now)
        recent = bars_to_frame(bars) if bars else pd.DataFrame()
        if local is None or local.empty:
            return recent  # too short to trade; on_hour says so
        if not recent.empty and recent.index[0] - local.index[-1] > MAX_GAP:
            start, end = local.index[-1].date(), recent.index[0].date() + timedelta(days=1)
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    fill, _ = await download_dukascopy_h1(client, inst.symbol, start, end)
                if fill:
                    write_bars(history_path(data_dir, "dukascopy", inst.symbol, "H1"), fill)
                    local = pd.concat([local, bars_to_frame(fill)])
                    local = local[~local.index.duplicated(keep="last")].sort_index()
                await log_event(self.e.db, "INFO", "runner", f"Filled {len(fill)} hourly bars of {inst.symbol} from Dukascopy")
            except Exception as ex:
                log.warning("Dukascopy top-up for %s failed: %s", inst.symbol, ex)
            if recent.index[0] - local.index[-1] > MAX_GAP:
                raise RuntimeError(f"history gap {local.index[-1]:%Y-%m-%d} to {recent.index[0]:%Y-%m-%d} could not be filled")
        if recent.empty:
            return local
        return pd.concat([local[local.index < recent.index[0]], recent])

    async def _h1(self, inst, now: datetime) -> pd.DataFrame:
        adapter = self.e.adapters[inst.venue]
        cached = self._cache.get(inst.symbol)
        if cached is None or cached.empty:
            if inst.venue is Venue.CAPITAL:
                df = await self._seed_from_file(inst, now)
            else:
                bars = await adapter.history(inst, "H1", now - timedelta(days=HISTORY_DAYS), now)
                df = bars_to_frame(bars) if bars else pd.DataFrame()
        else:
            bars = await adapter.history(inst, "H1", now - timedelta(days=REFRESH_DAYS), now)
            fresh = bars_to_frame(bars) if bars else pd.DataFrame()
            df = cached if fresh.empty else pd.concat([cached[cached.index < fresh.index[0]], fresh])
        if df.empty:
            return df
        df = df[df.index >= pd.Timestamp(now) - pd.Timedelta(days=HISTORY_DAYS)]
        self._cache[inst.symbol] = df
        return df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)]  # completed bars only

    def _params(self, name: str, sym: str, settings) -> dict:
        strat = self.strategies[name]
        return {**strat.default_params, **settings.strategy_params.get(name, {}).get(sym, {})}

    def _frame(self, name: str, inst, h1: pd.DataFrame, params: dict) -> pd.DataFrame:
        strat = self.strategies[name]
        key = (name, inst.symbol, prep_key(strat, params), h1.index[-1])
        if key not in self._frames:
            self._frames[key] = prepare(strat, h1, params, inst.kind is AssetKind.CRYPTO_SPOT)
        return self._frames[key]

    def _enough_history(self, name: str, h1: pd.DataFrame) -> tuple[bool, str]:
        strat = self.strategies[name]
        days = getattr(strat, "min_history_days", None)
        if days is not None:
            span = (h1.index[-1] - h1.index[0]).days if len(h1) else 0
            return span >= days, f"{span} days of hourly history, {name} needs {days}"
        return len(h1) > WARMUP_BARS, f"only {len(h1)} H1 bars, need more than {WARMUP_BARS}"

    async def on_hour(self, now: datetime) -> list[str]:
        settings = await self.e.settings_store.load()
        enabled = {s: syms for s, syms in settings.strategies_enabled.items() if s in self.strategies and syms}
        actions: list[str] = []
        self._frames = {}
        symbols = sorted({sym for syms in enabled.values() for sym in syms} | await self._open_strategy_symbols())
        for sym in symbols:
            inst = self.e.instruments.get(sym)
            if inst is None or inst.venue not in self.e.adapters:
                continue
            try:
                h1 = await self._h1(inst, now)
            except Exception as ex:
                await log_event(self.e.db, "WARNING", "runner", f"No usable H1 history for {sym}: {ex}")
                continue
            if h1.empty:
                await log_event(self.e.db, "WARNING", "runner", f"No H1 bars for {sym}")
                continue
            actions += await self.manage_positions(sym, h1, settings, now)
            for name, syms in enabled.items():
                if sym not in syms:
                    continue
                ok, why = self._enough_history(name, h1)
                if not ok:
                    await log_event(self.e.db, "WARNING", "runner", f"{sym}: {why}; not trading it yet")
                    continue
                actions += await self._signal(name, inst, h1, settings, now)
        return actions

    async def _signal(self, name: str, inst, h1: pd.DataFrame, settings, now: datetime) -> list[str]:
        strat = self.strategies[name]
        params = self._params(name, inst.symbol, settings)
        frame = self._frame(name, inst, h1, params)
        if frame.empty:
            return []
        i = len(frame) - 1
        if "ready" in frame.columns and not bool(frame["ready"].iloc[i]):
            return []
        if not _fresh(frame, i, h1, now):
            return []  # this bar's moment has passed: the backtest would have entered at the next open
        sig = strat.signal_at(columns(frame), i, params, inst.kind is not AssetKind.CRYPTO_SPOT)
        if sig is None:
            return []
        bar_time = frame.index[i].to_pydatetime()
        spec = self.e.specs.get(inst.symbol)
        tick = spec.price_tick if spec else None
        intent = TradeIntent(
            symbol=inst.symbol, side=sig.side, entry=_px(sig.entry, tick), stop=_px(sig.stop, tick), target=_px(sig.target, tick),
            source=Source.STRATEGY, strategy=name, signal_time=bar_time, reason=sig.reason,
        )
        decision, outcome = await self.e.trade(intent)
        if outcome is not None and outcome.status is OrderStatus.FILLED and not outcome.duplicate:
            cid = client_order_id(intent.source, name, inst.symbol, sig.side, bar_time)
            async with self.e.db.sessions() as s, s.begin():
                if await s.get(PositionMetaRow, cid) is None:
                    s.add(PositionMetaRow(
                        client_order_id=cid, strategy=name, initial_risk=_dec(abs(sig.entry - sig.stop)),
                        trail_atr=_dec(sig.trail_atr), breakeven_at_r=_dec(sig.breakeven_at_r), max_bars=sig.max_bars or 0,
                        signal_bar=bar_time,
                    ))
            return [f"opened {sig.side.value} {inst.symbol} ({name})"]
        why = "; ".join(decision.reasons) or (outcome.error if outcome else "needs approval")
        return [f"{name} {sig.side.value} {inst.symbol} not opened: {why}"]

    async def _open_strategy_symbols(self) -> set[str]:
        async with self.e.db.sessions() as s:
            rows = (await s.execute(select(PositionRow.symbol).where(
                PositionRow.status == PositionStatus.OPEN.value, PositionRow.source == Source.STRATEGY.value
            ))).scalars().all()
        return set(rows)

    async def manage_positions(self, sym: str, h1: pd.DataFrame, settings=None, now: datetime | None = None) -> list[str]:
        """Breakeven, ATR trail, time exit and signal exit for open strategy positions on this symbol,
        on the strategy's own bars."""
        async with self.e.db.sessions() as s:
            rows = (await s.execute(select(PositionRow, PositionMetaRow).join(
                PositionMetaRow, PositionMetaRow.client_order_id == PositionRow.client_order_id
            ).where(PositionRow.symbol == sym, PositionRow.status == PositionStatus.OPEN.value))).all()
        if not rows:
            return []
        settings = settings or await self.e.settings_store.load()
        inst = self.e.instruments.get(sym)
        out = []
        for pos, meta in rows:
            strat = self.strategies.get(meta.strategy)
            if strat is None or inst is None:
                continue
            params = self._params(meta.strategy, sym, settings)
            frame = self._frame(meta.strategy, inst, h1, params)
            side = Side(pos.side)
            signal_bar = pd.Timestamp(meta.signal_bar if meta.signal_bar.tzinfo else meta.signal_bar.replace(tzinfo=UTC))
            since = frame[frame.index > signal_bar]  # bars after the signal bar = bars the position has lived through
            if since.empty:
                continue
            if meta.max_bars and len(since) >= meta.max_bars:
                if await self.e.execution.close(pos.id, f"time exit after {meta.max_bars} bars"):
                    out.append(f"closed {sym} {pos.client_order_id}: time exit")
                continue
            i = len(frame) - 1
            exit_signal = hasattr(strat, "exit_at") and _fresh(frame, i, h1, now or datetime.now(UTC)) and strat.exit_at(columns(frame), i, side, params)
            if exit_signal:
                if await self.e.execution.close(pos.id, f"{meta.strategy} exit signal"):
                    out.append(f"closed {sym} {pos.client_order_id}: exit signal")
                continue
            best = float(since["close"].max() if side is Side.BUY else since["close"].min())
            current = float(pos.stop)
            atr_now = float(frame["atr"].iloc[-1])
            new = trailed_stop(side, float(pos.entry_price), current, float(meta.initial_risk), best, atr_now,
                               None if meta.trail_atr is None else float(meta.trail_atr),
                               None if meta.breakeven_at_r is None else float(meta.breakeven_at_r),
                               getattr(strat, "trail_activate_r", 1.0))
            spec = self.e.specs.get(sym)
            new_px = _px(new, spec.price_tick if spec else None)
            if (side is Side.BUY and new_px > pos.stop) or (side is Side.SELL and new_px < pos.stop):
                adapter = self.e.adapters[Venue(pos.venue)]
                if await adapter.modify_protection(journal_to_broker(pos), new_px, None):
                    async with self.e.db.sessions() as s, s.begin():
                        row = await s.get(PositionRow, pos.id)
                        row.stop = new_px
                    await log_event(self.e.db, "INFO", "runner", f"Stop tightened on {sym}", {"cid": pos.client_order_id, "from": current, "to": new_px})
                    out.append(f"tightened {sym} stop {current} -> {new_px}")
        return out
