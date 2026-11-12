"""Capital.com DEMO adapter (REST + WebSocket).

Only the demo REST endpoint is defined here. Stops and targets are attached to
the position at the broker, so they keep working if this engine stops.

API facts used (open-api.capital.com):
- POST /api/v1/session with X-CAP-API-KEY; tokens CST + X-SECURITY-TOKEN expire
  after 10 minutes without use.
- 10 requests/s per user; position/order creation 1 per 0.1 s; session 1/s.
- POST /positions returns a dealReference; the outcome must be read from
  GET /confirms/{dealReference}. There is no client-supplied idempotency key,
  so duplicate protection is done in the journal (see execution.py).
- WebSocket: max 40 epics per subscription, ping at least every 10 minutes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
from websockets.asyncio.client import connect as ws_connect

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
    Side,
    Venue,
)
from .base import TIMEFRAME_SECONDS, Backoff, BrokerError, MinInterval

log = logging.getLogger(__name__)

DEMO_REST = "https://demo-api-capital.backend-capital.com"
STREAM_URL = "wss://api-streaming-capital.backend-capital.com/connect"

RESOLUTION = {
    "M1": "MINUTE",
    "M5": "MINUTE_5",
    "M15": "MINUTE_15",
    "M30": "MINUTE_30",
    "H1": "HOUR",
    "H4": "HOUR_4",
    "D1": "DAY",
}
SESSION_IDLE_LIMIT = 9 * 60  # renew before Capital.com's 10-minute expiry
WS_PING_EVERY = 5 * 60


def _dec(v) -> Decimal | None:
    return None if v is None else Decimal(str(v))


def _parse_utc(s: str) -> datetime:
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fmt).replace(tzinfo=UTC)


def _fmt_time(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S")


class CapitalComAdapter:
    venue = Venue.CAPITAL

    def __init__(
        self,
        api_key: str,
        identifier: str,
        password: str,
        instruments: list[Instrument],
        client: httpx.AsyncClient | None = None,
        stream_url: str = STREAM_URL,
        confirm_attempts: int = 10,
        confirm_delay: float = 0.5,
    ):
        self._api_key = api_key
        self._identifier = identifier
        self._password = password
        self._http = client or httpx.AsyncClient(base_url=DEMO_REST, timeout=15.0)
        self._stream_url = stream_url
        self._cst: str | None = None
        self._xst: str | None = None
        self._last_used = 0.0
        self._account_id: str | None = None
        self._session_lock = asyncio.Lock()
        self._general = MinInterval(0.11)  # stays under 10 req/s
        self._orders = MinInterval(0.12)  # stays under 1 per 0.1 s
        self._session_rl = MinInterval(1.1)
        self._by_epic = {i.broker_symbol: i for i in instruments if i.venue is Venue.CAPITAL}
        self._specs: dict[str, MarketSpec] = {}
        self._confirm_attempts = confirm_attempts
        self._confirm_delay = confirm_delay
        self.session_renewals = 0
        self.history_gaps: list[tuple] = []

    def add_instrument(self, inst: Instrument) -> None:
        if inst.venue is Venue.CAPITAL:
            self._by_epic[inst.broker_symbol] = inst

    # ------------------------------------------------------------------ session
    async def connect(self) -> None:
        await self._login()

    async def close(self) -> None:
        await self._http.aclose()

    async def _login(self) -> None:
        await self._session_rl.wait()
        r = await self._http.post(
            "/api/v1/session",
            headers={"X-CAP-API-KEY": self._api_key},
            json={"identifier": self._identifier, "password": self._password, "encryptedPassword": False},
        )
        if r.status_code != 200:
            raise BrokerError(f"Capital.com login failed ({r.status_code}): {_error_text(r)}", status=r.status_code)
        self._cst = r.headers.get("CST")
        self._xst = r.headers.get("X-SECURITY-TOKEN")
        if not self._cst or not self._xst:
            raise BrokerError("Capital.com login returned no session tokens")
        body = r.json()
        self._account_id = body.get("currentAccountId")
        self._last_used = time.monotonic()
        self.session_renewals += 1

    def invalidate_session(self) -> None:
        """Test hook: forget tokens so the next call must log in again."""
        self._cst = self._xst = None

    async def _ensure_session(self) -> None:
        async with self._session_lock:
            if self._cst is None or time.monotonic() - self._last_used > SESSION_IDLE_LIMIT:
                await self._login()

    async def _request(self, method: str, path: str, *, params=None, json_body=None, order: bool = False) -> dict:
        for attempt in range(3):
            await self._ensure_session()
            await (self._orders if order else self._general).wait()
            r = await self._http.request(
                method,
                path,
                params=params,
                json=json_body,
                headers={"CST": self._cst or "", "X-SECURITY-TOKEN": self._xst or ""},
            )
            self._last_used = time.monotonic()
            if r.status_code == 401 and attempt == 0:
                async with self._session_lock:
                    await self._login()
                continue
            if r.status_code == 429:
                await asyncio.sleep(1.0 + attempt)
                continue
            if r.status_code >= 500:
                raise BrokerError(f"Capital.com {method} {path} server error {r.status_code}", status=r.status_code, retryable=True)
            if r.status_code >= 400:
                raise BrokerError(f"Capital.com {method} {path} failed ({r.status_code}): {_error_text(r)}", status=r.status_code)
            return r.json() if r.content else {}
        raise BrokerError(f"Capital.com {method} {path} failed after retries", retryable=True)

    async def ping(self) -> bool:
        body = await self._request("GET", "/api/v1/ping")
        return body.get("status") == "OK"

    # ------------------------------------------------------------------ reads
    async def account(self) -> AccountSnapshot:
        body = await self._request("GET", "/api/v1/accounts")
        accounts = body.get("accounts", [])
        acct = next((a for a in accounts if a.get("accountId") == self._account_id), None)
        acct = acct or next((a for a in accounts if a.get("preferred")), None) or (accounts[0] if accounts else None)
        if acct is None:
            raise BrokerError("Capital.com returned no accounts")
        b = acct["balance"]
        balance = Decimal(str(b["balance"]))
        pnl = Decimal(str(b.get("profitLoss", 0)))
        return AccountSnapshot(
            venue=self.venue,
            currency=acct.get("currency", "USD"),
            balance=balance,
            equity=balance + pnl,
            available=Decimal(str(b.get("available", balance))),
        )

    async def market_details(self, inst: Instrument) -> dict:
        return await self._request("GET", f"/api/v1/markets/{inst.broker_symbol}")

    async def market_spec(self, inst: Instrument) -> MarketSpec:
        body = await self.market_details(inst)
        rules = body.get("dealingRules", {})
        snap = body.get("snapshot", {})
        instrument = body.get("instrument", {})
        dp = int(snap.get("decimalPlacesFactor", 5))
        spec = MarketSpec(
            min_size=Decimal(str(rules.get("minDealSize", {}).get("value", 1))),
            size_step=Decimal(str(rules.get("minSizeIncrement", {}).get("value", 1))),
            price_tick=Decimal(1).scaleb(-dp),
            contract_multiplier=Decimal(str(instrument.get("lotSize", 1))),
            tradeable=snap.get("marketStatus") == "TRADEABLE",
        )
        self._specs[inst.symbol] = spec
        return spec

    async def quote(self, inst: Instrument) -> Quote:
        body = await self.market_details(inst)
        snap = body["snapshot"]
        return Quote(inst.symbol, Decimal(str(snap["bid"])), Decimal(str(snap["offer"])), datetime.now(UTC))

    async def history(self, inst: Instrument, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        """Fetch in windows of at most 1,000 bars so no request can be truncated by the API's cap."""
        res = RESOLUTION[timeframe]
        window = timedelta(seconds=TIMEFRAME_SECONDS[timeframe] * 1000)
        bars: dict[datetime, Bar] = {}
        cursor = start
        while cursor < end:
            stop = min(end, cursor + window)
            try:
                body = await self._request(
                    "GET",
                    f"/api/v1/prices/{inst.broker_symbol}",
                    params={"resolution": res, "max": 1000, "from": _fmt_time(cursor), "to": _fmt_time(stop)},
                )
            except BrokerError as e:
                # Capital.com answers 400/404 for windows older than it keeps (or with no prices,
                # e.g. a weekend): skip the window and carry on rather than lose the whole download.
                if e.status in (400, 404):
                    self.history_gaps.append((cursor, stop, str(e)))
                    cursor = stop
                    continue
                raise
            for p in body.get("prices", []):
                t = _parse_utc(p.get("snapshotTimeUTC") or p["snapshotTime"])
                bars[t] = Bar(
                    inst.symbol,
                    timeframe,
                    t,
                    _mid(p["openPrice"]),
                    _mid(p["highPrice"]),
                    _mid(p["lowPrice"]),
                    _mid(p["closePrice"]),
                    Decimal(str(p.get("lastTradedVolume", 0))),
                )
            cursor = stop
        return [bars[k] for k in sorted(bars)]

    async def list_positions(self) -> list[BrokerPosition]:
        body = await self._request("GET", "/api/v1/positions")
        out: list[BrokerPosition] = []
        for item in body.get("positions", []):
            p, m = item.get("position", {}), item.get("market", {})
            inst = self._by_epic.get(m.get("epic"))
            symbol = inst.symbol if inst else m.get("epic", "?")
            created = p.get("createdDateUTC") or p.get("createdDate")
            out.append(
                BrokerPosition(
                    venue=self.venue,
                    symbol=symbol,
                    side=Side(p["direction"]),
                    size=Decimal(str(p["size"])),
                    entry_price=Decimal(str(p["level"])),
                    stop=_dec(p.get("stopLevel")),
                    target=_dec(p.get("profitLevel")),
                    broker_position_id=str(p["dealId"]),
                    opened_at=_parse_utc(created) if created else None,
                    unrealised_pnl=_dec(p.get("upl")),
                )
            )
        return out

    # ------------------------------------------------------------------ trading
    async def _confirm(self, deal_reference: str) -> dict | None:
        for _ in range(self._confirm_attempts):
            try:
                body = await self._request("GET", f"/api/v1/confirms/{deal_reference}")
                if body.get("dealStatus"):
                    return body
            except BrokerError as e:
                if e.status not in (404,):
                    raise
            await asyncio.sleep(self._confirm_delay)
        return None

    def _price(self, inst: Instrument, value: Decimal) -> float:
        spec = self._specs.get(inst.symbol)
        if spec:
            value = (value / spec.price_tick).to_integral_value() * spec.price_tick
        return float(value)

    async def open_position(self, plan: OrderPlan) -> OpenResult:
        inst, intent = plan.instrument, plan.intent
        body = {
            "epic": inst.broker_symbol,
            "direction": intent.side.value,
            "size": float(plan.size),
            "guaranteedStop": False,
            "stopLevel": self._price(inst, intent.stop),
        }
        if intent.target is not None:
            body["profitLevel"] = self._price(inst, intent.target)
        resp = await self._request("POST", "/api/v1/positions", json_body=body, order=True)
        ref = resp.get("dealReference")
        if not ref:
            return OpenResult(ok=False, error="no dealReference returned", raw=resp)
        conf = await self._confirm(ref)
        if conf is None:
            return OpenResult(ok=False, error=f"unconfirmed deal {ref}", raw={"dealReference": ref})
        if conf.get("dealStatus") != "ACCEPTED":
            return OpenResult(ok=False, error=f"rejected: {conf.get('reason') or conf.get('status')}", raw=conf)
        deal_id = next(
            (d["dealId"] for d in conf.get("affectedDeals", []) if d.get("status") in ("OPENED", "OPEN")),
            conf.get("dealId"),
        )
        result = OpenResult(
            ok=True,
            broker_position_id=str(deal_id),
            fill_price=_dec(conf.get("level")),
            filled_size=_dec(conf.get("size")) or plan.size,
            raw=conf,
        )
        result.protected = await self._verify_stop(str(deal_id), intent.stop)
        return result

    async def _verify_stop(self, deal_id: str, stop: Decimal) -> bool:
        for pos in await self.list_positions():
            if pos.broker_position_id == deal_id:
                return pos.stop is not None
        return False

    async def find_order(self, plan: OrderPlan, since: datetime, known_ids: set[str]) -> OpenResult | None:
        """After an ambiguous failure: did our order open anyway? Match on epic, side, size, time."""
        for pos in await self.list_positions():
            if (
                pos.symbol == plan.instrument.symbol
                and pos.side is plan.intent.side
                and pos.size == plan.size
                and pos.broker_position_id not in known_ids
                and (pos.opened_at is None or pos.opened_at >= since - timedelta(seconds=10))
            ):
                return OpenResult(
                    ok=True,
                    broker_position_id=pos.broker_position_id,
                    fill_price=pos.entry_price,
                    filled_size=pos.size,
                    protected=pos.stop is not None,
                )
        return None

    async def modify_protection(self, pos: BrokerPosition, stop: Decimal | None, target: Decimal | None) -> bool:
        inst = next((i for i in self._by_epic.values() if i.symbol == pos.symbol), None)
        body: dict = {}
        if stop is not None:
            body["stopLevel"] = self._price(inst, stop) if inst else float(stop)
        if target is not None:
            body["profitLevel"] = self._price(inst, target) if inst else float(target)
        resp = await self._request("PUT", f"/api/v1/positions/{pos.broker_position_id}", json_body=body, order=True)
        ref = resp.get("dealReference")
        conf = await self._confirm(ref) if ref else None
        return bool(conf and conf.get("dealStatus") == "ACCEPTED")

    async def close_position(self, pos: BrokerPosition) -> bool:
        resp = await self._request("DELETE", f"/api/v1/positions/{pos.broker_position_id}", order=True)
        ref = resp.get("dealReference")
        conf = await self._confirm(ref) if ref else None
        return bool(conf and conf.get("dealStatus") == "ACCEPTED")

    async def exit_info(self, pos: BrokerPosition) -> ExitInfo:
        """Best effort: the position is gone, so it closed (stop, target or manual)."""
        price = None
        try:
            body = await self._request(
                "GET",
                "/api/v1/history/activity",
                params={"dealId": pos.broker_position_id, "lastPeriod": 86400, "detailed": "true"},
            )
            # Activities are newest first; the newest one with a level is the closing deal.
            for a in body.get("activities", []):
                level = (a.get("details") or {}).get("level")
                if level is not None:
                    price = _dec(level)
                    break
        except BrokerError:
            pass
        return ExitInfo("closed", price, "closed at broker (stop, target or manual)")

    # ------------------------------------------------------------------ streaming
    async def stream(self, instruments: list[Instrument]) -> AsyncIterator[Quote | Bar]:
        """Quotes for up to 40 epics, reconnecting forever with backoff."""
        epics = [i.broker_symbol for i in instruments if i.venue is Venue.CAPITAL][:40]
        backoff = Backoff()
        self.stream_connects = 0
        while True:
            try:
                await self._ensure_session()
                async with ws_connect(self._stream_url, ping_interval=None, open_timeout=15) as ws:
                    self.stream_connects += 1
                    await ws.send(self._ws_msg("marketData.subscribe", {"epics": epics}))
                    backoff.reset()
                    next_ping = time.monotonic() + WS_PING_EVERY
                    while True:
                        timeout = max(1.0, next_ping - time.monotonic())
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                        except TimeoutError:
                            await ws.send(self._ws_msg("ping", None))
                            next_ping = time.monotonic() + WS_PING_EVERY
                            continue
                        msg = json.loads(raw)
                        if msg.get("destination") == "quote":
                            p = msg.get("payload", {})
                            inst = self._by_epic.get(p.get("epic"))
                            if inst and p.get("bid") is not None and p.get("ofr") is not None:
                                ts = datetime.fromtimestamp(p.get("timestamp", time.time() * 1000) / 1000, UTC)
                                yield Quote(inst.symbol, Decimal(str(p["bid"])), Decimal(str(p["ofr"])), ts)
                        elif msg.get("status") == "ERROR":
                            log.warning("Capital.com stream error: %s", msg.get("payload"))
                            if "session" in json.dumps(msg.get("payload", "")).lower():
                                self.invalidate_session()
                                break
            except asyncio.CancelledError:
                raise
            except Exception as e:  # network drop, server close, auth expiry
                delay = backoff.next()
                log.warning("Capital.com stream disconnected (%s); reconnecting in %.0fs", type(e).__name__, delay)
                await asyncio.sleep(delay)

    def _ws_msg(self, destination: str, payload: dict | None) -> str:
        msg = {"destination": destination, "correlationId": str(int(time.time() * 1000)), "cst": self._cst, "securityToken": self._xst}
        if payload is not None:
            msg["payload"] = payload
        return json.dumps(msg)


def _mid(price_obj: dict) -> Decimal:
    bid, ask = price_obj.get("bid"), price_obj.get("ask")
    if bid is None:
        return Decimal(str(ask))
    if ask is None:
        return Decimal(str(bid))
    return (Decimal(str(bid)) + Decimal(str(ask))) / 2


def _error_text(r: httpx.Response) -> str:
    try:
        return str(r.json().get("errorCode", r.text[:200]))
    except Exception:
        return r.text[:200]
