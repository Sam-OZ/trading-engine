"""Live strategy runner: once per hour, the same strategy code the backtester used.

At each hour close (plus a short grace period so the broker has the bar):
1. For every instrument with an enabled strategy, fetch the latest completed H1 bars.
2. prepare() + signal_at(last bar) -> a TradeIntent through the normal risk + execution path.
   The signal time is the bar's open time, so the client order id is the same however
   many times this hour is processed: a restart cannot double an order.
3. Manage open strategy positions exactly as the backtester does: breakeven at 1R,
   ATR trail, time exit. Stops only ever tighten, and the new stop is placed at the broker.

Nothing runs until `strategies_enabled` lists a strategy for a symbol (set by
`tradeengine gate2-apply` after Gate 2). An empty setting means the runner is idle.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from sqlalchemy import select

from .db import PositionMetaRow, PositionRow
from .execution import journal_to_broker
from .history import bars_to_frame
from .ids import client_order_id
from .indicators import atr as atr_indicator
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

log = logging.getLogger(__name__)

# The 4-hour EMA200 needs a long run-in before it stops depending on where the data starts:
# 240 days gives ~1,000 four-hour bars, so live values match the backtest's to well under a pip.
HISTORY_DAYS = 240
REFRESH_DAYS = 3  # after the first load, only the last few days are fetched each hour
GRACE_SECONDS = 20


def _dec(x: float | None) -> Decimal | None:
    return None if x is None else Decimal(repr(float(x)))


def _px(x: float | None, tick: Decimal | None) -> Decimal | None:
    """A price as the broker will hold it: rounded to the instrument's tick."""
    d = _dec(x)
    return d if d is None or not tick else round_to_tick(d, tick)


class StrategyRunner:
    def __init__(self, engine, strategies: dict | None = None):
        self.e = engine
        self.strategies = strategies or STRATEGIES
        self._cache: dict[str, pd.DataFrame] = {}

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

    async def _h1(self, inst, now: datetime) -> pd.DataFrame:
        adapter = self.e.adapters[inst.venue]
        cached = self._cache.get(inst.symbol)
        since = now - timedelta(days=REFRESH_DAYS if cached is not None and len(cached) else HISTORY_DAYS)
        bars = await adapter.history(inst, "H1", since, now)
        fresh = bars_to_frame(bars) if bars else pd.DataFrame()
        df = fresh if cached is None or cached.empty else pd.concat([cached[cached.index < fresh.index[0]] if len(fresh) else cached, fresh])
        if df.empty:
            return df
        df = df[df.index >= pd.Timestamp(now) - pd.Timedelta(days=HISTORY_DAYS)]
        self._cache[inst.symbol] = df
        return df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp(now)]  # completed bars only

    async def on_hour(self, now: datetime) -> list[str]:
        settings = await self.e.settings_store.load()
        enabled = {s: syms for s, syms in settings.strategies_enabled.items() if s in self.strategies and syms}
        actions: list[str] = []
        symbols = sorted({sym for syms in enabled.values() for sym in syms} | await self._open_strategy_symbols())
        for sym in symbols:
            inst = self.e.instruments.get(sym)
            if inst is None or inst.venue not in self.e.adapters:
                continue
            try:
                h1 = await self._h1(inst, now)
            except Exception as ex:
                await log_event(self.e.db, "WARNING", "runner", f"No H1 history for {sym}: {ex}")
                continue
            if len(h1) <= WARMUP_BARS:
                await log_event(self.e.db, "WARNING", "runner", f"Only {len(h1)} H1 bars for {sym}; need more than {WARMUP_BARS}")
                continue
            actions += await self.manage_positions(sym, h1)
            for name, syms in enabled.items():
                if sym in syms:
                    actions += await self._signal(name, inst, h1, settings)
        return actions

    async def _signal(self, name: str, inst, h1: pd.DataFrame, settings) -> list[str]:
        strat = self.strategies[name]
        params = {**strat.default_params, **settings.strategy_params.get(name, {}).get(inst.symbol, {})}
        frame = strat.prepare(h1, params)
        i = len(frame) - 1
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
                        trail_atr=_dec(sig.trail_atr), breakeven_at_r=_dec(sig.breakeven_at_r), max_bars=sig.max_bars, signal_bar=bar_time,
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

    async def manage_positions(self, sym: str, h1: pd.DataFrame) -> list[str]:
        """Breakeven, ATR trail and time exit for open strategy positions on this symbol."""
        async with self.e.db.sessions() as s:
            rows = (await s.execute(select(PositionRow, PositionMetaRow).join(
                PositionMetaRow, PositionMetaRow.client_order_id == PositionRow.client_order_id
            ).where(PositionRow.symbol == sym, PositionRow.status == PositionStatus.OPEN.value))).all()
        if not rows:
            return []
        atr_now = float(atr_indicator(h1["high"], h1["low"], h1["close"], 14).iloc[-1])
        out = []
        for pos, meta in rows:
            side = Side(pos.side)
            signal_bar = pd.Timestamp(meta.signal_bar if meta.signal_bar.tzinfo else meta.signal_bar.replace(tzinfo=UTC))
            since = h1[h1.index > signal_bar]  # bars after the signal bar = bars the position has lived through
            if since.empty:
                continue
            if len(since) >= meta.max_bars:
                if await self.e.execution.close(pos.id, f"time exit after {meta.max_bars} bars"):
                    out.append(f"closed {sym} {pos.client_order_id}: time exit")
                continue
            best = float(since["close"].max() if side is Side.BUY else since["close"].min())
            current = float(pos.stop)
            new = trailed_stop(side, float(pos.entry_price), current, float(meta.initial_risk), best, atr_now,
                               None if meta.trail_atr is None else float(meta.trail_atr),
                               None if meta.breakeven_at_r is None else float(meta.breakeven_at_r))
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

