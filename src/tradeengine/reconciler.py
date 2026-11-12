"""Reconciler: makes the journal and the brokers agree, every N seconds and on start-up.

Rules:
- Journal open + broker open: copy the broker's live stop/target into the journal.
  If the broker shows NO stop, re-apply the journal stop; if that fails, close it.
- Journal open + broker gone: ask the adapter what happened. Closed -> record it.
  Coins still held with protection cancelled (Binance) -> re-protect or sell.
- Broker position the journal does not know: alert only. It was not opened by
  this engine (for example a manual demo trade), so the engine does not touch it.
- Orders left SUBMITTED/UNKNOWN (crash or timeout mid-send): look them up and
  record the fill, or mark them rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from .brokers.base import BrokerAdapter
from .db import Database, OrderRow, PositionRow
from .execution import ExecutionService, journal_to_broker
from .models import (
    DEFAULT_INSTRUMENTS,
    BrokerPosition,
    Instrument,
    OrderPlan,
    OrderStatus,
    PositionStatus,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from .state import log_event

log = logging.getLogger(__name__)


@dataclass
class ReconcileReport:
    matched: int = 0
    closed_at_broker: list[str] = field(default_factory=list)
    reprotected: list[str] = field(default_factory=list)
    force_closed: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)
    orders_settled: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.reprotected or self.force_closed or self.external or self.errors)


class Reconciler:
    def __init__(
        self,
        db: Database,
        adapters: dict[Venue, BrokerAdapter],
        execution: ExecutionService,
        instruments: dict[str, Instrument] | None = None,
    ):
        self.db = db
        self.adapters = adapters
        self.execution = execution
        self.instruments = instruments or DEFAULT_INSTRUMENTS

    async def run_once(self) -> ReconcileReport:
        report = ReconcileReport()
        await self._settle_orders(report)
        for venue, adapter in self.adapters.items():
            try:
                await self._venue(venue, adapter, report)
            except Exception as e:  # one venue failing must not stop the other
                report.errors.append(f"{venue}: {e}")
                log.exception("reconcile failed for %s", venue)
        if not report.clean:
            await log_event(self.db, "WARNING", "reconciler", "Reconcile found issues", report.__dict__)
        return report

    async def _venue(self, venue: Venue, adapter: BrokerAdapter, report: ReconcileReport) -> None:
        broker = await adapter.list_positions()
        async with self.db.sessions() as s:
            journal = (
                await s.execute(select(PositionRow).where(PositionRow.venue == venue.value, PositionRow.status == PositionStatus.OPEN.value))
            ).scalars().all()

        def key_j(p: PositionRow) -> str:
            return p.client_order_id if venue is Venue.BINANCE else p.broker_position_id

        def key_b(p: BrokerPosition) -> str:
            return (p.client_order_id or p.broker_position_id) if venue is Venue.BINANCE else p.broker_position_id

        by_key = {key_b(b): b for b in broker}
        seen: set[str] = set()

        for jp in journal:
            k = key_j(jp)
            bp = by_key.get(k)
            if bp is not None:
                seen.add(k)
                report.matched += 1
                if bp.stop is None:
                    await self._reprotect_or_close(adapter, jp, bp, report)
                else:
                    await self._sync_levels(jp.id, bp)
                continue
            info = await adapter.exit_info(journal_to_broker(jp))
            if info.status == "closed":
                await self.execution.mark_closed(jp.id, info.exit_price, info.reason)
                report.closed_at_broker.append(f"{jp.symbol} ({info.reason})")
            elif info.status == "unprotected":
                await self._reprotect_or_close(adapter, jp, journal_to_broker(jp), report)
            else:
                report.errors.append(f"{jp.symbol} {jp.client_order_id}: {info.reason}")

        for k, bp in by_key.items():
            if k not in seen:
                report.external.append(f"{venue} {bp.symbol} {bp.side} {bp.size} id={bp.broker_position_id}")

    async def _reprotect_or_close(self, adapter: BrokerAdapter, jp: PositionRow, bp: BrokerPosition, report: ReconcileReport) -> None:
        try:
            if jp.stop is not None and await adapter.modify_protection(bp, jp.stop, jp.target):
                report.reprotected.append(f"{jp.symbol} {jp.client_order_id}")
                await log_event(self.db, "ERROR", "reconciler", "Position had no stop; re-applied", {"cid": jp.client_order_id})
                return
        except Exception as e:
            log.error("re-protect failed for %s: %s", jp.client_order_id, e)
        ok = await adapter.close_position(bp)
        await self.execution.mark_closed(jp.id, None, "closed by reconciler: no stop")
        report.force_closed.append(f"{jp.symbol} {jp.client_order_id} closed={ok}")
        await log_event(self.db, "CRITICAL", "reconciler", "Unprotected position closed", {"cid": jp.client_order_id, "ok": ok})

    async def _sync_levels(self, pos_id: int, bp: BrokerPosition) -> None:
        async with self.db.sessions() as s, s.begin():
            row = await s.get(PositionRow, pos_id)
            if row is None:
                return
            if bp.stop is not None and row.stop != bp.stop:
                row.stop = bp.stop
            if bp.target is not None and row.target != bp.target:
                row.target = bp.target
            if bp.protection_id and row.protection_id != bp.protection_id:
                row.protection_id = bp.protection_id

    async def _settle_orders(self, report: ReconcileReport) -> None:
        cutoff = datetime.now(UTC) - timedelta(seconds=30)
        async with self.db.sessions() as s:
            rows = (
                await s.execute(
                    select(OrderRow).where(
                        OrderRow.status.in_([OrderStatus.SUBMITTED.value, OrderStatus.UNKNOWN.value]),
                    )
                )
            ).scalars().all()
        for o in rows:
            created = o.created_at if o.created_at.tzinfo else o.created_at.replace(tzinfo=UTC)
            if created > cutoff:
                continue
            inst = self.instruments.get(o.symbol)
            adapter = self.adapters.get(Venue(o.venue))
            if inst is None or adapter is None:
                continue
            plan = OrderPlan(
                o.client_order_id,
                inst,
                TradeIntent(o.symbol, Side(o.side), o.entry_ref, o.stop, o.target, Source(o.source), o.strategy, created),
                o.size,
                o.risk_amount,
            )
            try:
                found = await adapter.find_order(plan, created, await self.execution.known_broker_ids())
            except Exception as e:
                report.errors.append(f"settle {o.client_order_id}: {e}")
                continue
            if found and found.ok:
                await self.execution.record_fill(plan, found)
                report.orders_settled.append(f"{o.client_order_id} filled")
            elif datetime.now(UTC) - created > timedelta(minutes=10):
                await self.execution.set_order_status(o.client_order_id, OrderStatus.REJECTED, "not found at broker after 10 min")
                report.orders_settled.append(f"{o.client_order_id} rejected")
