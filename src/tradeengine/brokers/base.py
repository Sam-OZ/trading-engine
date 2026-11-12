"""The interface every broker adapter implements, plus shared helpers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Protocol

from ..models import (
    AccountSnapshot,
    Bar,
    BrokerPosition,
    ExitInfo,
    Instrument,
    MarketSpec,
    OpenResult,
    OrderPlan,
    Quote,
    Venue,
)

TIMEFRAME_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}


class BrokerError(Exception):
    """A broker refused or failed a request. Message never contains credentials."""

    def __init__(self, message: str, status: int | None = None, code: str | int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.code = code
        self.retryable = retryable


class BrokerAdapter(Protocol):
    venue: Venue

    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def account(self) -> AccountSnapshot: ...
    async def market_spec(self, inst: Instrument) -> MarketSpec: ...
    async def quote(self, inst: Instrument) -> Quote: ...
    async def history(self, inst: Instrument, timeframe: str, start: datetime, end: datetime) -> list[Bar]: ...
    async def open_position(self, plan: OrderPlan) -> OpenResult: ...
    async def find_order(self, plan: OrderPlan, since: datetime, known_ids: set[str]) -> OpenResult | None: ...
    async def list_positions(self) -> list[BrokerPosition]: ...
    async def modify_protection(self, pos: BrokerPosition, stop: object, target: object) -> bool: ...
    async def close_position(self, pos: BrokerPosition) -> bool: ...
    async def exit_info(self, pos: BrokerPosition) -> ExitInfo: ...
    def stream(self, instruments: list[Instrument]) -> AsyncIterator[Quote | Bar]: ...


class MinInterval:
    """Async limiter: at most one call per `interval` seconds (shared by all callers)."""

    def __init__(self, interval: float):
        self.interval = interval
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


class Backoff:
    """Exponential backoff for reconnect loops: 1, 2, 4 ... capped, reset on success."""

    def __init__(self, base: float = 1.0, cap: float = 60.0):
        self.base, self.cap, self.n = base, cap, 0

    def next(self) -> float:
        delay = min(self.cap, self.base * (2**self.n))
        self.n += 1
        return delay

    def reset(self) -> None:
        self.n = 0
