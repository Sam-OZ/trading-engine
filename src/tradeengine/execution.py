"""Execution: turns a risk-approved OrderPlan into a protected broker position.

Duplicate prevention is layered:
1. The client order ID is derived from the signal (ids.py), and the journal
   has a UNIQUE constraint on it. The row is written BEFORE anything is sent,
   so a second submit of the same signal stops at the database.
2. If a send fails ambiguously (timeout, 5xx), we never blindly resend. We ask
   the broker whether the order exists; if the answer is still unclear the
   order is marked UNKNOWN and the reconciler settles it.

A position must end this function protected by a broker-side stop, or closed.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import httpx
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from .brokers.base import BrokerAdapter, BrokerError
from .db import Database, OrderRow, PositionRow
from .models import (
    BrokerPosition,
    OpenResult,
    OrderPlan,
    OrderStatus,
    PositionStatus,
    Side,
    Venue,
)
from .state import EngineState, log_event

log = logging.getLogger(__name__)

AMBIGUOUS = (httpx.TransportError, TimeoutError, asyncio.TimeoutError)


@dataclass
class ExecutionOutcome:
    status: OrderStatus
    duplicate: bool = False
    position_id: int | None = None
    error: str | None = None


class ExecutionService:
    def __init__(self, db: Database, adapters: dict[Venue, BrokerAdapter], state: EngineState, recheck_delay: float = 2.0):
        self.db = db
        self.adapters = adapters
        self.state = state
        self.recheck_delay = recheck_delay

    async def submit(self, plan: OrderPlan) -> ExecutionOutcome:
        halted, why = await self.state.is_halted()
        if halted:
            return ExecutionOutcome(OrderStatus.REJECTED, error=f"kill switch engaged: {why}")

        inst, intent = plan.instrument, plan.intent
        try:
            async with self.db.sessions() as s, s.begin():
                s.add(
                    OrderRow(
                        client_order_id=plan.client_order_id,
                        venue=inst.venue.value,
                        symbol=inst.symbol,
                        side=intent.side.value,
                        size=plan.size,
                        source=intent.source.value,
                        strategy=intent.strategy,
                        entry_ref=intent.entry,
                        stop=intent.stop,
                        target=intent.target,
                        risk_amount=plan.risk_amount,
                        status=OrderStatus.SUBMITTED.value,
                        reason=intent.reason,
                    )
                )
        except IntegrityError:
            existing = await self._order(plan.client_order_id)
            await log_event(self.db, "WARNING", "execution", "Duplicate order blocked", {"cid": plan.client_order_id})
            return ExecutionOutcome(OrderStatus(existing.status) if existing else OrderStatus.UNKNOWN, duplicate=True)

        adapter = self.adapters[inst.venue]
        since = datetime.now(UTC)
        try:
            result = await adapter.open_position(plan)
        except AMBIGUOUS as e:
            result = await self._resolve_ambiguous(plan, since, f"{type(e).__name__}")
        except BrokerError as e:
            if e.retryable:
                result = await self._resolve_ambiguous(plan, since, str(e))
            else:
                await self.set_order_status(plan.client_order_id, OrderStatus.REJECTED, error=str(e))
                await log_event(self.db, "WARNING", "execution", f"Order rejected: {e}", {"cid": plan.client_order_id})
                return ExecutionOutcome(OrderStatus.REJECTED, error=str(e))

        if result is None:
            await self.set_order_status(plan.client_order_id, OrderStatus.UNKNOWN, error="send failed; not found at broker yet")
            await log_event(self.db, "ERROR", "execution", "Order outcome unknown; reconciler will settle", {"cid": plan.client_order_id})
            return ExecutionOutcome(OrderStatus.UNKNOWN, error="outcome unknown")

        if not result.ok:
            status = OrderStatus.UNKNOWN if (result.error or "").startswith("unconfirmed") else OrderStatus.REJECTED
            await self.set_order_status(plan.client_order_id, status, error=result.error)
            await log_event(self.db, "WARNING", "execution", f"Order not opened: {result.error}", {"cid": plan.client_order_id})
            return ExecutionOutcome(status, error=result.error)

        pos_id = await self.record_fill(plan, result)
        if not result.protected:
            await self._enforce_protection(plan, result, pos_id)
        return ExecutionOutcome(OrderStatus.FILLED, position_id=pos_id)

    async def _resolve_ambiguous(self, plan: OrderPlan, since: datetime, why: str) -> OpenResult | None:
        await log_event(self.db, "WARNING", "execution", f"Ambiguous send ({why}); checking broker", {"cid": plan.client_order_id})
        await asyncio.sleep(self.recheck_delay)
        try:
            return await self.adapters[plan.instrument.venue].find_order(plan, since, await self.known_broker_ids())
        except Exception as e:  # still unclear: leave it to the reconciler
            log.warning("find_order failed: %s", e)
            return None

    async def record_fill(self, plan: OrderPlan, result: OpenResult) -> int:
        intent = plan.intent
        async with self.db.sessions() as s, s.begin():
            order = (await s.execute(select(OrderRow).where(OrderRow.client_order_id == plan.client_order_id))).scalar_one()
            order.status = OrderStatus.FILLED.value
            order.broker_position_id = result.broker_position_id
            order.fill_price = result.fill_price
            order.error = None
            pos = PositionRow(
                client_order_id=plan.client_order_id,
                venue=plan.instrument.venue.value,
                symbol=plan.instrument.symbol,
                side=intent.side.value,
                size=result.filled_size or plan.size,
                entry_price=result.fill_price or intent.entry,
                stop=intent.stop,
                target=intent.target,
                broker_position_id=result.broker_position_id or "",
                protection_id=result.protection_id,
                source=intent.source.value,
                strategy=intent.strategy,
                risk_amount=plan.risk_amount,
                status=PositionStatus.OPEN.value,
            )
            s.add(pos)
            await s.flush()
            pos_id = pos.id
        await log_event(
            self.db,
            "INFO",
            "execution",
            f"Opened {intent.side.value} {plan.size} {plan.instrument.symbol}",
            {"cid": plan.client_order_id, "fill": result.fill_price, "stop": intent.stop, "target": intent.target, "source": intent.source},
        )
        return pos_id

    async def _enforce_protection(self, plan: OrderPlan, result: OpenResult, pos_id: int) -> None:
        """Broker accepted the order but no stop is visible: set it, or close the position."""
        adapter = self.adapters[plan.instrument.venue]
        bp = BrokerPosition(
            venue=plan.instrument.venue,
            symbol=plan.instrument.symbol,
            side=plan.intent.side,
            size=result.filled_size or plan.size,
            entry_price=result.fill_price or plan.intent.entry,
            stop=None,
            target=None,
            broker_position_id=result.broker_position_id or "",
            protection_id=result.protection_id,
            client_order_id=plan.client_order_id,
        )
        try:
            if await adapter.modify_protection(bp, plan.intent.stop, plan.intent.target):
                await log_event(self.db, "WARNING", "execution", "Stop was missing after fill; re-applied", {"cid": plan.client_order_id})
                return
        except Exception as e:
            log.error("re-applying stop failed: %s", e)
        closed = await adapter.close_position(bp)
        await self.mark_closed(pos_id, None, "closed: no broker-side stop")
        await log_event(
            self.db, "CRITICAL", "execution", "Position had no stop and was closed", {"cid": plan.client_order_id, "closed": closed}
        )

    async def mark_closed(self, pos_id: int, exit_price: Decimal | None, reason: str) -> None:
        async with self.db.sessions() as s, s.begin():
            pos = await s.get(PositionRow, pos_id)
            if pos is None or pos.status == PositionStatus.CLOSED.value:
                return
            pos.status = PositionStatus.CLOSED.value
            pos.closed_at = datetime.now(UTC)
            pos.exit_price = exit_price
            pos.exit_reason = reason[:64]
            if exit_price is not None:
                pos.realised_pnl = (exit_price - pos.entry_price) * pos.size * Side(pos.side).sign
            symbol, side, cid, pnl = pos.symbol, pos.side, pos.client_order_id, pos.realised_pnl
        # every close goes through here (engine exit, broker stop seen by the reconciler, emergency): one event each
        await log_event(
            self.db, "INFO", "execution",
            f"Closed {side} {symbol}: {reason}" + (f"; P&L {pnl:+.2f}" if pnl is not None else ""),
            {"cid": cid, "exit": exit_price, "pnl": pnl},
        )

    async def close(self, pos_id: int, reason: str) -> bool:
        async with self.db.sessions() as s:
            pos = await s.get(PositionRow, pos_id)
        if pos is None or pos.status != PositionStatus.OPEN.value:
            return False
        adapter = self.adapters[Venue(pos.venue)]
        bp = journal_to_broker(pos)
        ok = await adapter.close_position(bp)
        if ok:
            exit_price = None
            try:  # the broker's own record of the closing fill, so the journal holds a real exit price and P&L
                info = await adapter.exit_info(bp)
                exit_price = info.exit_price if info.status == "closed" else None
            except Exception as e:
                log.warning("exit price lookup failed for %s: %s", pos.client_order_id, e)
            await self.mark_closed(pos_id, exit_price, reason)
        return ok

    async def known_broker_ids(self) -> set[str]:
        async with self.db.sessions() as s:
            rows = (await s.execute(select(PositionRow.broker_position_id))).scalars().all()
        return set(rows)

    async def _order(self, cid: str) -> OrderRow | None:
        async with self.db.sessions() as s:
            return (await s.execute(select(OrderRow).where(OrderRow.client_order_id == cid))).scalar_one_or_none()

    async def set_order_status(self, cid: str, status: OrderStatus, error: str | None = None) -> None:
        async with self.db.sessions() as s, s.begin():
            row = (await s.execute(select(OrderRow).where(OrderRow.client_order_id == cid))).scalar_one()
            row.status = status.value
            row.error = error


def journal_to_broker(pos: PositionRow) -> BrokerPosition:
    return BrokerPosition(
        venue=Venue(pos.venue),
        symbol=pos.symbol,
        side=Side(pos.side),
        size=pos.size,
        entry_price=pos.entry_price,
        stop=pos.stop,
        target=pos.target,
        broker_position_id=pos.broker_position_id,
        protection_id=pos.protection_id,
        client_order_id=pos.client_order_id,
        opened_at=(pos.opened_at if pos.opened_at is None or pos.opened_at.tzinfo else pos.opened_at.replace(tzinfo=UTC)),
    )
