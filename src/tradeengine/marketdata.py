"""Live market data: tick freshness, 1-minute bars from quotes, and resampling."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from .brokers.base import TIMEFRAME_SECONDS
from .models import Bar, Quote


class TickTracker:
    """Last quote per symbol, for staleness checks and currency conversion."""

    def __init__(self) -> None:
        self.last: dict[str, Quote] = {}
        self._mono: dict[str, float] = {}

    def update(self, q: Quote) -> None:
        self.last[q.symbol] = q
        self._mono[q.symbol] = time.monotonic()

    def age_seconds(self) -> dict[str, float]:
        now = time.monotonic()
        return {s: now - t for s, t in self._mono.items()}

    def mids(self) -> dict[str, Decimal]:
        return {s: q.mid for s, q in self.last.items()}


class MinuteBarBuilder:
    """Builds 1-minute mid-price bars from quotes; returns each bar when its minute ends.

    Minutes with no quotes produce no bar (no invented prices).
    """

    def __init__(self) -> None:
        self._open: dict[str, dict] = {}

    def on_quote(self, q: Quote) -> Bar | None:
        minute = q.ts.astimezone(UTC).replace(second=0, microsecond=0)
        px = q.mid
        cur = self._open.get(q.symbol)
        closed = None
        if cur is not None and minute > cur["t"]:
            closed = Bar(q.symbol, "M1", cur["t"], cur["o"], cur["h"], cur["l"], cur["c"], Decimal(cur["n"]))
            cur = None
        if cur is None:
            self._open[q.symbol] = {"t": minute, "o": px, "h": px, "l": px, "c": px, "n": 1}
        elif minute == cur["t"]:
            cur["h"] = max(cur["h"], px)
            cur["l"] = min(cur["l"], px)
            cur["c"] = px
            cur["n"] += 1
        return closed


def bucket_start(t: datetime, timeframe: str) -> datetime:
    secs = TIMEFRAME_SECONDS[timeframe]
    epoch = int(t.astimezone(UTC).timestamp())
    return datetime.fromtimestamp(epoch - epoch % secs, UTC)


def resample(bars: list[Bar], timeframe: str) -> list[Bar]:
    """Aggregate bars (any lower timeframe, sorted by time) into `timeframe`, UTC-aligned."""
    out: list[Bar] = []
    cur: dict | None = None
    for b in bars:
        start = bucket_start(b.open_time, timeframe)
        if cur is None or start != cur["t"]:
            if cur is not None:
                out.append(Bar(b.symbol, timeframe, cur["t"], cur["o"], cur["h"], cur["l"], cur["c"], cur["v"]))
            cur = {"t": start, "o": b.open, "h": b.high, "l": b.low, "c": b.close, "v": b.volume}
        else:
            cur["h"] = max(cur["h"], b.high)
            cur["l"] = min(cur["l"], b.low)
            cur["c"] = b.close
            cur["v"] += b.volume
    if cur is not None:
        out.append(Bar(bars[-1].symbol, timeframe, cur["t"], cur["o"], cur["h"], cur["l"], cur["c"], cur["v"]))
    return out


def complete_only(bars: list[Bar], timeframe: str, now: datetime) -> list[Bar]:
    """Drop the last bar if its period has not finished yet (no look-ahead)."""
    span = timedelta(seconds=TIMEFRAME_SECONDS[timeframe])
    return [b for b in bars if b.open_time + span <= now]


class BarStore:
    """In-memory rolling window of bars per (symbol, timeframe)."""

    def __init__(self, maxlen: int = 2000) -> None:
        self._bars: dict[tuple[str, str], deque[Bar]] = defaultdict(lambda: deque(maxlen=maxlen))

    def add(self, bar: Bar) -> None:
        dq = self._bars[(bar.symbol, bar.timeframe)]
        if dq and dq[-1].open_time >= bar.open_time:
            if dq[-1].open_time == bar.open_time:
                dq[-1] = bar
            return
        dq.append(bar)

    def get(self, symbol: str, timeframe: str) -> list[Bar]:
        return list(self._bars[(symbol, timeframe)])
