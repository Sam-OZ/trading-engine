"""Persistent engine state and the kill switch.

The kill switch is engaged by ANY of: the file data/KILL existing, or the
`halted` flag in the database. Either source alone stops new entries, so the
switch still works if the database is down (touch the file) or the server's
disk is read-only (dashboard sets the flag).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from .db import Database, EventRow, StateRow


class EngineState:
    def __init__(self, db: Database, kill_file: Path):
        self.db = db
        self.kill_file = kill_file

    async def _get(self, key: str) -> dict | None:
        async with self.db.sessions() as s:
            row = await s.get(StateRow, key)
            return row.value if row else None

    async def _set(self, key: str, value: dict) -> None:
        async with self.db.sessions() as s, s.begin():
            row = await s.get(StateRow, key)
            if row is None:
                s.add(StateRow(key=key, value=value))
            else:
                row.value = value

    # --- kill switch -------------------------------------------------------
    async def is_halted(self) -> tuple[bool, str]:
        if self.kill_file.exists():
            reason = self.kill_file.read_text().strip() or "kill file present"
            return True, reason
        v = await self._get("halted")
        if v and v.get("halted"):
            return True, v.get("reason", "halted")
        return False, ""

    async def engage_kill_switch(self, reason: str, by: str) -> None:
        self.kill_file.parent.mkdir(parents=True, exist_ok=True)
        self.kill_file.write_text(f"{reason} (by {by} at {datetime.now(UTC).isoformat()})")
        await self._set("halted", {"halted": True, "reason": reason, "by": by})
        await log_event(self.db, "CRITICAL", "killswitch", f"Kill switch engaged: {reason}", {"by": by})

    async def release_kill_switch(self, by: str) -> None:
        if self.kill_file.exists():
            self.kill_file.unlink()
        await self._set("halted", {"halted": False, "reason": "", "by": by})
        await log_event(self.db, "WARNING", "killswitch", "Kill switch released", {"by": by})

    # --- equity markers (per venue, in that venue's account currency) -------
    async def start_of_day_equity(self, venue: str, day: str, current: Decimal) -> Decimal:
        """Return the equity recorded at the first look of `day` (UTC), recording it if new."""
        key = f"sod_equity:{venue}:{day}"
        v = await self._get(key)
        if v is None:
            await self._set(key, {"equity": str(current)})
            return current
        return Decimal(v["equity"])

    async def update_peak_equity(self, venue: str, equity: Decimal) -> Decimal:
        key = f"peak_equity:{venue}"
        v = await self._get(key)
        peak = Decimal(v["equity"]) if v else None
        if peak is None or equity > peak:
            await self._set(key, {"equity": str(equity)})
            return equity
        return peak


async def log_event(db: Database, level: str, component: str, message: str, data: dict | None = None) -> None:
    safe = json.loads(json.dumps(data, default=str)) if data else None
    async with db.sessions() as s, s.begin():
        s.add(EventRow(level=level, component=component, message=message, data=safe))
