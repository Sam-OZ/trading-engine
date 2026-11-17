"""Telegram alerts: trades, problems and a daily summary, read from the engine's own event log.

The notifier does not hook into the trading code. Every decision the engine makes is already written
to the `events` table; this task reads new rows every 30 seconds and forwards the ones worth a phone
notification. A failed send is retried on the next pass (the bookmark only moves after a send works),
so a Telegram outage delays alerts but never loses them, and never affects trading.

Sent: opens and closes; WARNING from execution, reconciler, risk, kill switch, engine; every ERROR
and CRITICAL. The same text is sent at most once per `telegram_repeat_hours`. Not sent: routine INFO
(stop tightened, approvals) - those are in the daily summary and the log.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
from sqlalchemy import func, select

from .alerting import RECOVERED
from .db import EventRow, PositionRow, StateRow
from .models import PositionStatus

log = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    """A Telegram API call failed. The message never contains the bot token (Telegram puts it in the URL)."""


def _check(r: httpx.Response, what: str) -> None:
    if r.status_code != 200:
        raise TelegramError(f"Telegram {what} failed: HTTP {r.status_code}")

API = "https://api.telegram.org"
BOOKMARK = "telegram_last_event_id"
LAST_SUMMARY = "telegram_last_summary"
FORWARD_START = "forward_test_start"
WARN_COMPONENTS = {"execution", "reconciler", "risk", "killswitch", "engine", "emergency"}
ICON = {"CRITICAL": "🛑", "ERROR": "❗", "WARNING": "⚠️"}


def worth_sending(ev: EventRow) -> bool:
    if ev.level in ("ERROR", "CRITICAL"):
        return True
    if ev.level == "WARNING":
        return ev.component in WARN_COMPONENTS
    if ev.message.startswith(RECOVERED):  # the all-clear after an outage that was alerted
        return True
    return ev.component == "execution" and ev.message.startswith(("Opened", "Closed"))


def format_event(ev: EventRow) -> str:
    if ev.component == "execution" and ev.level == "INFO":
        icon = "🟢" if ev.message.startswith("Opened") else "🔵"
        return f"{icon} {ev.message}"
    if ev.message.startswith(RECOVERED):
        return f"✅ {ev.component}: {ev.message.removeprefix(RECOVERED)}"
    return f"{ICON.get(ev.level, '')} {ev.level} {ev.component}: {ev.message}".strip()


async def send(client: httpx.AsyncClient, token: str, chat_id: str, text: str) -> None:
    for i in range(0, len(text), 3900):  # Telegram's limit is 4,096 characters per message
        r = await client.post(f"{API}/bot{token}/sendMessage", json={"chat_id": chat_id, "text": text[i:i + 3900],
                                                                    "disable_web_page_preview": True})
        _check(r, "sendMessage")


async def find_chat_id(client: httpx.AsyncClient, token: str) -> tuple[str, str] | None:
    """The chat that most recently messaged the bot: (chat id, name). Send the bot any message first."""
    r = await client.get(f"{API}/bot{token}/getUpdates")
    _check(r, "getUpdates")
    for upd in reversed(r.json().get("result", [])):
        chat = (upd.get("message") or upd.get("edited_message") or {}).get("chat")
        if chat:
            return str(chat["id"]), chat.get("first_name") or chat.get("title") or chat.get("username") or ""
    return None


async def _state(db, key: str):
    async with db.sessions() as s:
        row = await s.get(StateRow, key)
        return row.value if row else None


async def _set_state(db, key: str, value) -> None:
    async with db.sessions() as s, s.begin():
        row = await s.get(StateRow, key)
        if row is None:
            s.add(StateRow(key=key, value=value))
        else:
            row.value = value


async def summary_text(engine, now: datetime) -> str:
    """Account, open positions, the last 24 hours and the forward test so far."""
    await engine.refresh_accounts()
    halted, why = await engine.state.is_halted()
    lines = [f"📊 Daily summary {now:%a %d %b %Y} (UTC {now:%H:%M})"]
    for v, a in engine.accounts.items():
        lines.append(f"{v.value}: equity {a.equity:,.2f} {a.currency} (available {a.available:,.2f})")
    lines.append(f"Kill switch: {'ENGAGED - ' + why if halted else 'off'}")
    async with engine.db.sessions() as s:
        open_rows = (await s.execute(select(PositionRow).where(PositionRow.status == PositionStatus.OPEN.value))).scalars().all()
        day = (await s.execute(select(PositionRow).where(PositionRow.closed_at >= now - timedelta(days=1)))).scalars().all()
        start = await s.get(StateRow, FORWARD_START)
        ft = []
        if start:
            since = datetime.fromisoformat(start.value["at"])
            ft = (await s.execute(select(PositionRow).where(
                PositionRow.status == PositionStatus.CLOSED.value, PositionRow.closed_at >= since,
                PositionRow.strategy.in_(start.value.get("strategies", [])),
            ))).scalars().all()
        n_events = (await s.execute(select(func.count(EventRow.id)).where(
            EventRow.ts >= now - timedelta(days=1), EventRow.level.in_(["ERROR", "CRITICAL"])))).scalar_one()
    if open_rows:
        lines.append("Open:")
        for p in open_rows:
            q = engine.ticks.last.get(p.symbol)
            u = f", unrealised {(q.mid - p.entry_price) * p.size * (1 if p.side == 'BUY' else -1):+.2f}" if q else ""
            lines.append(f"  {p.side} {p.size} {p.symbol} @ {p.entry_price} stop {p.stop} ({p.strategy}){u}")
    else:
        lines.append("Open: none")
    lines.append(f"Closed in the last 24 h: {len(day)}" + (
        "; P&L " + ", ".join(f"{p.symbol} {p.realised_pnl:+.2f}" for p in day if p.realised_pnl is not None) if day else ""))
    if start:
        pnl = [p.realised_pnl for p in ft if p.realised_pnl is not None]
        wins = sum((x for x in pnl if x > 0), Decimal(0))
        losses = -sum((x for x in pnl if x <= 0), Decimal(0))
        pf = f"{wins / losses:.2f}" if losses > 0 else ("n/a" if not pnl else "all wins")
        lines.append(f"Forward test since {start.value['at'][:10]} ({', '.join(start.value.get('strategies', []))}): "
                     f"{len(ft)} closed trades, P&L {sum(pnl, Decimal(0)):+.2f}, profit factor {pf}")
    try:  # round 4 Track A shadow tally, once its traders are frozen; never let it break the summary
        from .round4_follow import stats as follow_stats
        from .round4_follow import summary_line

        st = await follow_stats(engine.db, now)
        if st is not None:
            lines.append(summary_line(st))
    except Exception as e:
        log.warning("Track A summary line failed: %s", e)
    lines.append(f"Errors in the last 24 h: {n_events}")
    return "\n".join(lines)


class TelegramNotifier:
    def __init__(self, engine, token: str):
        self.e = engine
        self.token = token
        self._recent: dict[str, datetime] = {}

    async def run_forever(self, every: float = 30.0) -> None:
        async with httpx.AsyncClient(timeout=15) as client:
            while True:
                try:
                    await self.run_once(client, datetime.now(UTC))
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # never let alerts affect trading
                    log.warning("telegram notifier: %s", e)
                await asyncio.sleep(every)

    async def run_once(self, client: httpx.AsyncClient, now: datetime) -> int:
        settings = await self.e.settings_store.load()
        if not settings.telegram_alerts_enabled or not settings.telegram_chat_id:
            return 0
        chat = settings.telegram_chat_id
        last = await _state(self.e.db, BOOKMARK)
        if last is None:  # first start: begin from now, do not replay history
            async with self.e.db.sessions() as s:
                top = (await s.execute(select(func.max(EventRow.id)))).scalar_one() or 0
            await _set_state(self.e.db, BOOKMARK, {"id": top})
            return 0
        async with self.e.db.sessions() as s:
            rows = (await s.execute(select(EventRow).where(EventRow.id > last["id"]).order_by(EventRow.id).limit(200))).scalars().all()
        sent = 0
        window = timedelta(hours=settings.telegram_repeat_hours)
        for ev in rows:
            if worth_sending(ev):
                text = format_event(ev)
                key = f"{ev.level}|{ev.component}|{ev.message}"
                repeat = ev.level != "INFO" and key in self._recent and now - self._recent[key] < window
                if not repeat:
                    await send(client, self.token, chat, text)  # raises -> bookmark stays, retried next pass
                    self._recent[key] = now
                    sent += 1
            await _set_state(self.e.db, BOOKMARK, {"id": ev.id})
        hh, mm = (int(x) for x in settings.telegram_summary_utc.split(":"))
        due = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        done = await _state(self.e.db, LAST_SUMMARY)
        if now >= due and (done or {}).get("date") != now.date().isoformat():
            await send(client, self.token, chat, await summary_text(self.e, now))
            await _set_state(self.e.db, LAST_SUMMARY, {"date": now.date().isoformat()})
            sent += 1
        return sent
