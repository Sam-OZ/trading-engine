"""Quiet handling of brief outages.

A broker's HTTP 5xx, a rate limit or a dropped connection usually clears within a minute, and every
job that hits one simply runs again on its next pass. So these "transient" failures go to the server
log only. Telegram hears about them when the SAME job has kept failing for `transient_alert_minutes`
(admin setting, default 3), and once more when it works again. Any other failure alerts at once.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx

from .brokers.base import BrokerError
from .state import log_event

log = logging.getLogger(__name__)

RECOVERED = "Recovered: "


def is_transient(e: BaseException) -> bool:
    """A failure that retrying later is expected to fix: broker 5xx / rate limit, timeouts, dropped connections."""
    if isinstance(e, BrokerError):
        return e.retryable
    return isinstance(e, (httpx.TransportError, asyncio.TimeoutError, TimeoutError))


def duration(td: timedelta) -> str:
    mins = int(td.total_seconds() // 60)
    if mins < 1:
        return "under a minute"
    if mins < 120:
        return f"{mins} min"
    return f"{mins // 60} h {mins % 60} min"


@dataclass
class _Streak:
    since: datetime
    tries: int = 0
    alerted: bool = False


class FailureAlerts:
    def __init__(self, db, settings_store, clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.db = db
        self.settings_store = settings_store
        self.clock = clock
        self._streaks: dict[tuple[str, str], _Streak] = {}

    async def failed(self, job: str, e: BaseException, component: str = "engine") -> None:
        if not is_transient(e):
            await log_event(self.db, "ERROR", component, f"{job} failed: {e}")
            return
        now = self.clock()
        st = self._streaks.setdefault((component, job), _Streak(since=now))
        st.tries += 1
        log.warning("%s failed (try %d, retrying quietly): %s", job, st.tries, e)
        if st.alerted:
            return
        minutes = float((await self.settings_store.load()).transient_alert_minutes)
        if now - st.since >= timedelta(minutes=minutes):
            st.alerted = True
            await log_event(self.db, "ERROR", component,
                            f"{job} failing for {duration(now - st.since)} ({st.tries} tries): {e}")

    async def succeeded(self, job: str, component: str = "engine") -> None:
        st = self._streaks.pop((component, job), None)
        if st is None:
            return
        if st.alerted:
            await log_event(self.db, "INFO", component, f"{RECOVERED}{job} working again after {duration(self.clock() - st.since)}")
        else:
            log.info("%s working again after %d quiet failure(s)", job, st.tries)
