from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from tradeengine.db import Database
from tradeengine.models import (
    DEFAULT_INSTRUMENTS,
    AccountSnapshot,
    BrokerPosition,
    ExitInfo,
    MarketSpec,
    OpenResult,
    OrderPlan,
    Quote,
    Venue,
)
from tradeengine.state import EngineState

D = Decimal


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await database.create_all()
    yield database
    await database.close()


@pytest.fixture
def state(db, tmp_path):
    return EngineState(db, tmp_path / "KILL")


class FakeAdapter:
    """In-memory broker for execution / reconciler / engine tests."""

    def __init__(self, venue: Venue = Venue.CAPITAL, currency: str = "USD"):
        self.venue = venue
        self.currency = currency
        self.positions: dict[str, BrokerPosition] = {}
        self.next_id = 1
        self.open_calls = 0
        self.fail_open_with: Exception | None = None
        self.open_lands_anyway = False  # with fail_open_with: the order still opened
        self.protect = True
        self.modify_ok = True
        self.closed: list[str] = []
        self.exit = ExitInfo("closed", D("1.1"), "stop loss")
        self.prices: dict[str, Quote] = {}

    async def connect(self):
        pass

    async def close(self):
        pass

    async def account(self):
        return AccountSnapshot(self.venue, self.currency, D("10000"), D("10000"), D("9000"))

    async def market_spec(self, inst):
        if inst.venue is Venue.BINANCE:
            return MarketSpec(D("0.00001"), D("0.00001"), D("0.01"), D("5"))
        return MarketSpec(D("1"), D("1"), D("0.00001"))

    async def quote(self, inst):
        return self.prices.get(inst.symbol) or Quote(inst.symbol, D("1.0999"), D("1.1001"), datetime.now(UTC))

    async def history(self, *a):
        return []

    def _make(self, plan: OrderPlan) -> BrokerPosition:
        pid = f"D{self.next_id}"
        self.next_id += 1
        bp = BrokerPosition(
            self.venue,
            plan.instrument.symbol,
            plan.intent.side,
            plan.size,
            plan.intent.entry,
            plan.intent.stop if self.protect else None,
            plan.intent.target if self.protect else None,
            pid,
            client_order_id=plan.client_order_id,
            opened_at=datetime.now(UTC),
        )
        self.positions[pid] = bp
        return bp

    async def open_position(self, plan):
        self.open_calls += 1
        if self.fail_open_with is not None:
            if self.open_lands_anyway:
                self._make(plan)
            raise self.fail_open_with
        bp = self._make(plan)
        return OpenResult(True, bp.broker_position_id, plan.intent.entry, plan.size, protected=bp.stop is not None)

    async def find_order(self, plan, since, known_ids):
        for bp in self.positions.values():
            if bp.client_order_id == plan.client_order_id and bp.broker_position_id not in known_ids:
                return OpenResult(True, bp.broker_position_id, bp.entry_price, bp.size, protected=bp.stop is not None)
        return None

    async def list_positions(self):
        return list(self.positions.values())

    async def modify_protection(self, pos, stop, target):
        if not self.modify_ok:
            return False
        bp = self.positions.get(pos.broker_position_id)
        if bp:
            bp.stop = stop if stop is not None else bp.stop
            bp.target = target if target is not None else bp.target
        return True

    async def close_position(self, pos):
        self.positions.pop(pos.broker_position_id, None)
        self.closed.append(pos.broker_position_id)
        return True

    async def exit_info(self, pos):
        return self.exit

    async def stream(self, instruments):
        if False:
            yield None


@pytest.fixture
def fake():
    return FakeAdapter()


EURUSD = DEFAULT_INSTRUMENTS["EUR/USD"]
USDJPY = DEFAULT_INSTRUMENTS["USD/JPY"]
GBPUSD = DEFAULT_INSTRUMENTS["GBP/USD"]
BTCUSDT = DEFAULT_INSTRUMENTS["BTC/USDT"]
