"""Execution and reconciler against an in-memory broker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import httpx
from conftest import EURUSD, FakeAdapter
from sqlalchemy import select

from tradeengine.brokers.base import BrokerError
from tradeengine.db import OrderRow, PositionRow
from tradeengine.execution import ExecutionService
from tradeengine.ids import client_order_id
from tradeengine.models import (
    BrokerPosition,
    ExitInfo,
    OrderPlan,
    OrderStatus,
    Side,
    Source,
    TradeIntent,
    Venue,
)
from tradeengine.reconciler import Reconciler


def plan(t: datetime | None = None) -> OrderPlan:
    t = t or datetime(2026, 11, 11, 14, 0, tzinfo=UTC)
    i = TradeIntent("EUR/USD", Side.BUY, D("1.1000"), D("1.0976"), D("1.1060"), Source.STRATEGY, "trend", t)
    return OrderPlan(client_order_id(i.source, i.strategy, i.symbol, i.side, t), EURUSD, i, D("20833"), D("50"))


async def rows(db, model):
    async with db.sessions() as s:
        return (await s.execute(select(model))).scalars().all()


async def test_submit_fills_and_journals(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    out = await ex.submit(plan())
    assert out.status is OrderStatus.FILLED and out.position_id
    (o,) = await rows(db, OrderRow)
    (p,) = await rows(db, PositionRow)
    assert o.status == "filled" and p.status == "open" and p.stop == D("1.0976")


async def test_duplicate_submit_never_reaches_broker(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    again = await ex.submit(plan())
    assert again.duplicate and fake.open_calls == 1 and len(fake.positions) == 1


async def test_timeout_but_order_landed_is_recorded_once(db, state):
    fake = FakeAdapter()
    fake.fail_open_with = httpx.ReadTimeout("slow")
    fake.open_lands_anyway = True
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    out = await ex.submit(plan())
    assert out.status is OrderStatus.FILLED
    assert fake.open_calls == 1 and len(fake.positions) == 1  # no blind resend


async def test_timeout_and_not_found_is_unknown_not_resent(db, state):
    fake = FakeAdapter()
    fake.fail_open_with = httpx.ConnectTimeout("down")
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    out = await ex.submit(plan())
    assert out.status is OrderStatus.UNKNOWN and fake.open_calls == 1
    (o,) = await rows(db, OrderRow)
    assert o.status == "unknown"


async def test_broker_rejection(db, state):
    fake = FakeAdapter()
    fake.fail_open_with = BrokerError("insufficient margin", status=400)
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    out = await ex.submit(plan())
    assert out.status is OrderStatus.REJECTED and "margin" in out.error


async def test_missing_stop_is_reapplied(db, state):
    fake = FakeAdapter()
    fake.protect = False
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    (bp,) = fake.positions.values()
    assert bp.stop == D("1.0976")


async def test_missing_stop_that_cannot_be_set_closes_position(db, state):
    fake = FakeAdapter()
    fake.protect = False
    fake.modify_ok = False
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    assert fake.positions == {} and fake.closed
    (p,) = await rows(db, PositionRow)
    assert p.status == "closed" and "no broker-side stop" in p.exit_reason


async def test_kill_switch_blocks_execution(db, state, fake):
    await state.engage_kill_switch("test", "t")
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    out = await ex.submit(plan())
    assert out.status is OrderStatus.REJECTED and fake.open_calls == 0


# ---------------------------------------------------------------- reconciler
async def test_reconcile_clean(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.matched == 1 and rep.clean


async def test_reconcile_records_broker_close(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    fake.positions.clear()  # stop hit at broker
    fake.exit = ExitInfo("closed", D("1.0976"), "stop loss")
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.closed_at_broker
    (p,) = await rows(db, PositionRow)
    assert p.status == "closed" and p.exit_price == D("1.0976")
    assert p.realised_pnl == (D("1.0976") - D("1.1000")) * D("20833")


async def test_reconcile_reprotects_or_closes_stopless_position(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    await ex.submit(plan())
    (bp,) = fake.positions.values()
    bp.stop = None  # someone removed the stop at the broker
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.reprotected and bp.stop == D("1.0976")
    bp.stop = None
    fake.modify_ok = False
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.force_closed and fake.positions == {}


async def test_reconcile_leaves_external_positions_alone(db, state, fake):
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    fake.positions["MANUAL1"] = BrokerPosition(Venue.CAPITAL, "EUR/USD", Side.SELL, D("1000"), D("1.1"), None, None, "MANUAL1")
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.external and "MANUAL1" in fake.positions and not fake.closed


async def test_reconcile_settles_unknown_order(db, state):
    fake = FakeAdapter()
    fake.fail_open_with = httpx.ConnectTimeout("down")
    ex = ExecutionService(db, {Venue.CAPITAL: fake}, state, recheck_delay=0)
    p = plan()
    await ex.submit(p)
    # The order did land after all; make the journal row old enough to settle.
    fake.fail_open_with = None
    fake._make(p)
    async with db.sessions() as s, s.begin():
        o = (await s.execute(select(OrderRow))).scalar_one()
        o.created_at = datetime.now(UTC) - timedelta(minutes=2)
    rep = await Reconciler(db, {Venue.CAPITAL: fake}, ex).run_once()
    assert rep.orders_settled == [f"{p.client_order_id} filled"]
    (o,) = await rows(db, OrderRow)
    assert o.status == "filled" and len(await rows(db, PositionRow)) == 1


async def test_one_venue_failing_does_not_stop_the_other(db, state, fake):
    class Broken(FakeAdapter):
        async def list_positions(self):
            raise BrokerError("down", retryable=True)

    ex = ExecutionService(db, {Venue.CAPITAL: fake, Venue.BINANCE: Broken(Venue.BINANCE)}, state, recheck_delay=0)
    await ex.submit(plan())
    rep = await Reconciler(db, {Venue.CAPITAL: fake, Venue.BINANCE: Broken(Venue.BINANCE)}, ex).run_once()
    assert rep.matched == 1 and rep.errors
