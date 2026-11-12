"""Binance Spot DEMO MODE adapter.

Two separate HTTP clients on purpose:
- `trade`: the Demo Mode endpoint. The ONLY place API keys are sent.
- `market`: the live public market-data endpoint, unauthenticated. Signals use
  real prices because Demo Mode prices are only "realistic"; no key ever goes here.

Spot is long-only. After the market BUY fills, the stop loss and take profit
go on the exchange as one OCO SELL (LIMIT_MAKER above, STOP_LOSS below), so the
position stays protected if this engine stops. If protection cannot be placed,
the position is sold immediately.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from decimal import Decimal
from urllib.parse import urlencode

import httpx
from websockets.asyncio.client import connect as ws_connect

from ..ids import leg_id
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
from ..sizing import round_down, round_to_tick
from .base import Backoff, BrokerError, MinInterval

log = logging.getLogger(__name__)

DEMO_TRADE_REST = "https://demo-api.binance.com"
LIVE_PUBLIC_REST = "https://api.binance.com"
LIVE_PUBLIC_WS = "wss://stream.binance.com:9443/stream"

INTERVAL = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
_LEG_RE = re.compile(r"^(te-[0-9a-f]{20})-(tp|sl|oco)(\d*)$")


def _d(v) -> Decimal:
    return Decimal(str(v)).normalize() if v is not None else Decimal(0)


def _s(v: Decimal) -> str:
    """Decimal to a plain string Binance accepts (no exponent)."""
    return format(v.normalize(), "f")


class BinanceSpotDemoAdapter:
    venue = Venue.BINANCE

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        instruments: list[Instrument],
        trade_client: httpx.AsyncClient | None = None,
        market_client: httpx.AsyncClient | None = None,
        stream_url: str = LIVE_PUBLIC_WS,
    ):
        self._key = api_key
        self._secret = api_secret.encode()
        self._trade = trade_client or httpx.AsyncClient(base_url=DEMO_TRADE_REST, timeout=15.0)
        self._market = market_client or httpx.AsyncClient(base_url=LIVE_PUBLIC_REST, timeout=15.0)
        self._stream_url = stream_url
        self._rl = MinInterval(0.1)
        self._time_offset_ms = 0
        self._by_symbol = {i.broker_symbol: i for i in instruments if i.venue is Venue.BINANCE}
        self._by_canon = {i.symbol: i for i in self._by_symbol.values()}
        self._specs: dict[str, MarketSpec] = {}

    # ------------------------------------------------------------------ plumbing
    async def connect(self) -> None:
        r = await self._trade.get("/api/v3/time")
        _check(r, "/api/v3/time")
        self._time_offset_ms = int(r.json()["serverTime"]) - int(time.time() * 1000)

    async def close(self) -> None:
        await self._trade.aclose()
        await self._market.aclose()

    def sign(self, params: dict) -> str:
        query = urlencode(params)
        sig = hmac.new(self._secret, query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    async def _signed(self, method: str, path: str, params: dict) -> dict | list:
        for attempt in range(3):
            await self._rl.wait()
            p = dict(params)
            p["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            p["recvWindow"] = 5000
            r = await self._trade.request(method, f"{path}?{self.sign(p)}", headers={"X-MBX-APIKEY": self._key})
            if r.status_code in (429, 418):
                await asyncio.sleep(float(r.headers.get("Retry-After", 2 + attempt)))
                continue
            return _check(r, path)
        raise BrokerError(f"Binance {path} rate limited", retryable=True)

    async def _public(self, path: str, params: dict | None = None) -> dict | list:
        await self._rl.wait()
        r = await self._market.get(path, params=params)
        return _check(r, path)

    # ------------------------------------------------------------------ reads
    async def account(self) -> AccountSnapshot:
        body = await self._signed("GET", "/api/v3/account", {"omitZeroBalances": "true"})
        bal = {b["asset"]: (_d(b["free"]), _d(b["locked"])) for b in body.get("balances", [])}
        usdt_free, usdt_locked = bal.get("USDT", (Decimal(0), Decimal(0)))
        equity = usdt_free + usdt_locked
        for inst in self._by_symbol.values():
            free, locked = bal.get(inst.base, (Decimal(0), Decimal(0)))
            if free + locked > 0:
                q = await self.quote(inst)
                equity += (free + locked) * q.bid
        return AccountSnapshot(self.venue, "USDT", usdt_free + usdt_locked, equity, usdt_free)

    async def balances(self) -> dict[str, Decimal]:
        body = await self._signed("GET", "/api/v3/account", {"omitZeroBalances": "true"})
        return {b["asset"]: _d(b["free"]) + _d(b["locked"]) for b in body.get("balances", [])}

    async def market_spec(self, inst: Instrument) -> MarketSpec:
        body = await self._demo_public("/api/v3/exchangeInfo", {"symbol": inst.broker_symbol})
        sym = body["symbols"][0]
        f = {x["filterType"]: x for x in sym["filters"]}
        notional = f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", "0")
        spec = MarketSpec(
            min_size=_d(f["LOT_SIZE"]["minQty"]),
            size_step=_d(f["LOT_SIZE"]["stepSize"]),
            price_tick=_d(f["PRICE_FILTER"]["tickSize"]),
            min_notional=_d(notional),
            tradeable=sym.get("status") == "TRADING",
        )
        self._specs[inst.symbol] = spec
        return spec

    async def _demo_public(self, path: str, params: dict) -> dict:
        """Unsigned GET on the DEMO endpoint (exchange filters match live exactly there)."""
        await self._rl.wait()
        r = await self._trade.get(path, params=params)
        return _check(r, path)

    async def quote(self, inst: Instrument) -> Quote:
        body = await self._public("/api/v3/ticker/bookTicker", {"symbol": inst.broker_symbol})
        return Quote(inst.symbol, _d(body["bidPrice"]), _d(body["askPrice"]), datetime.now(UTC))

    async def history(self, inst: Instrument, timeframe: str, start: datetime, end: datetime) -> list[Bar]:
        interval = INTERVAL[timeframe]
        out: list[Bar] = []
        cursor = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        while cursor < end_ms:
            rows = await self._public(
                "/api/v3/klines",
                {"symbol": inst.broker_symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000},
            )
            if not rows:
                break
            for k in rows:
                out.append(
                    Bar(inst.symbol, timeframe, datetime.fromtimestamp(k[0] / 1000, UTC), _d(k[1]), _d(k[2]), _d(k[3]), _d(k[4]), _d(k[5]))
                )
            nxt = int(rows[-1][0]) + 1
            if nxt <= cursor or len(rows) < 1000:
                break
            cursor = nxt
        return out

    # ------------------------------------------------------------------ trading
    async def open_position(self, plan: OrderPlan) -> OpenResult:
        inst, intent = plan.instrument, plan.intent
        if intent.side is not Side.BUY:
            return OpenResult(ok=False, error="spot is long-only")
        spec = self._specs.get(inst.symbol) or await self.market_spec(inst)
        qty = round_down(plan.size, spec.size_step)
        order = await self._signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": inst.broker_symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": _s(qty),
                "newClientOrderId": plan.client_order_id,
                "newOrderRespType": "FULL",
            },
        )
        return await self._protect_fill(plan, order, spec)

    async def _protect_fill(self, plan: OrderPlan, order: dict, spec: MarketSpec, cap_qty: Decimal | None = None) -> OpenResult:
        inst, intent = plan.instrument, plan.intent
        if order.get("status") not in ("FILLED", "PARTIALLY_FILLED"):
            return OpenResult(ok=False, error=f"order status {order.get('status')}", raw=order)
        executed = _d(order["executedQty"])
        avg = _d(order["cummulativeQuoteQty"]) / executed if executed else Decimal(0)
        base_fee = sum((_d(f["commission"]) for f in order.get("fills", []) if f.get("commissionAsset") == inst.base), Decimal(0))
        net = executed - base_fee
        if cap_qty is not None:
            net = min(net, cap_qty)
        net = round_down(net, spec.size_step)
        result = OpenResult(ok=True, broker_position_id=str(order["orderId"]), fill_price=avg, filled_size=net, raw=order)
        try:
            result.protection_id = await self._place_protection(inst, plan.client_order_id, net, intent.stop, intent.target, spec, 0)
            result.protected = True
        except BrokerError as e:
            log.error("Binance protection failed for %s: %s; selling immediately", plan.client_order_id, e)
            await self._market_sell(inst, net, leg_id(plan.client_order_id, "x"))
            return OpenResult(ok=False, error=f"protection failed, position sold: {e}", raw=order)
        return result

    async def _place_protection(
        self, inst: Instrument, cid: str, qty: Decimal, stop: Decimal, target: Decimal | None, spec: MarketSpec, version: int
    ) -> str:
        v = str(version) if version else ""
        stop_px = round_to_tick(stop, spec.price_tick)
        last = (await self.quote(inst)).bid
        if target is not None and round_to_tick(target, spec.price_tick) > last > stop_px:
            list_id = f"{cid}-oco{v}"
            await self._signed(
                "POST",
                "/api/v3/orderList/oco",
                {
                    "symbol": inst.broker_symbol,
                    "side": "SELL",
                    "quantity": _s(qty),
                    "listClientOrderId": list_id,
                    "aboveType": "LIMIT_MAKER",
                    "abovePrice": _s(round_to_tick(target, spec.price_tick)),
                    "aboveClientOrderId": f"{cid}-tp{v}",
                    "belowType": "STOP_LOSS",
                    "belowStopPrice": _s(stop_px),
                    "belowClientOrderId": f"{cid}-sl{v}",
                },
            )
            return list_id
        if not last > stop_px:
            raise BrokerError(f"price {last} already at or below stop {stop_px}")
        sl_id = f"{cid}-sl{v}"
        await self._signed(
            "POST",
            "/api/v3/order",
            {
                "symbol": inst.broker_symbol,
                "side": "SELL",
                "type": "STOP_LOSS",
                "quantity": _s(qty),
                "stopPrice": _s(stop_px),
                "newClientOrderId": sl_id,
            },
        )
        return sl_id

    async def _market_sell(self, inst: Instrument, qty: Decimal, cid: str) -> dict:
        return await self._signed(
            "POST",
            "/api/v3/order",
            {"symbol": inst.broker_symbol, "side": "SELL", "type": "MARKET", "quantity": _s(qty), "newClientOrderId": cid},
        )

    async def find_order(self, plan: OrderPlan, since: datetime, known_ids: set[str]) -> OpenResult | None:
        try:
            order = await self._signed(
                "GET", "/api/v3/order", {"symbol": plan.instrument.broker_symbol, "origClientOrderId": plan.client_order_id}
            )
        except BrokerError as e:
            if str(e.code) == "-2013":  # order does not exist
                return None
            raise
        if order.get("status") not in ("FILLED", "PARTIALLY_FILLED"):
            return None
        for p in await self.list_positions():
            if p.client_order_id == plan.client_order_id:
                return OpenResult(
                    ok=True, broker_position_id=str(order["orderId"]), filled_size=p.size, protection_id=p.protection_id, protected=True
                )
        # Filled but unprotected: protect it now (or sell). GET /order has no fill
        # details, so cap the quantity at what the account actually holds.
        spec = self._specs.get(plan.instrument.symbol) or await self.market_spec(plan.instrument)
        held = (await self.balances()).get(plan.instrument.base, Decimal(0))
        return await self._protect_fill(plan, order, spec, cap_qty=held)

    async def list_positions(self) -> list[BrokerPosition]:
        """Engine positions = our open protective orders (te-...-sl / -tp / -oco)."""
        out: dict[str, BrokerPosition] = {}
        for inst in self._by_symbol.values():
            orders = await self._signed("GET", "/api/v3/openOrders", {"symbol": inst.broker_symbol})
            for o in orders:
                m = _LEG_RE.match(o.get("clientOrderId", ""))
                if not m:
                    continue
                cid, leg = m.group(1), m.group(2)
                pos = out.setdefault(
                    cid,
                    BrokerPosition(
                        venue=self.venue,
                        symbol=inst.symbol,
                        side=Side.BUY,
                        size=_d(o["origQty"]) - _d(o.get("executedQty", 0)),
                        entry_price=Decimal(0),
                        stop=None,
                        target=None,
                        broker_position_id=cid,
                        client_order_id=cid,
                    ),
                )
                if leg == "sl":
                    pos.stop = _d(o.get("stopPrice"))
                elif leg == "tp":
                    pos.target = _d(o.get("price"))
                if o.get("orderListId", -1) != -1:
                    pos.protection_id = f"{cid}-oco{m.group(3)}"
                elif leg == "sl":
                    pos.protection_id = o["clientOrderId"]
        return list(out.values())

    async def _cancel_protection(self, inst: Instrument, protection_id: str) -> None:
        try:
            if "-oco" in protection_id:
                await self._signed("DELETE", "/api/v3/orderList", {"symbol": inst.broker_symbol, "listClientOrderId": protection_id})
            else:
                await self._signed("DELETE", "/api/v3/order", {"symbol": inst.broker_symbol, "origClientOrderId": protection_id})
        except BrokerError as e:
            if str(e.code) not in ("-2011", "-2013"):  # already gone
                raise

    async def _cancel_all_protection(self, inst: Instrument, cid: str) -> int:
        """Cancel every open protective order for this position, whatever the journal remembers.

        Looking at the live orders (not a stored id) means a stale protection id can
        never leave an old OCO working after the coins are sold.
        """
        orders = await self._signed("GET", "/api/v3/openOrders", {"symbol": inst.broker_symbol})
        lists, singles = set(), []
        for o in orders:
            m = _LEG_RE.match(o.get("clientOrderId", ""))
            if not m or m.group(1) != cid:
                continue
            if o.get("orderListId", -1) != -1:
                lists.add(o["orderListId"])
            else:
                singles.append(o["clientOrderId"])
        for list_id in lists:
            try:
                await self._signed("DELETE", "/api/v3/orderList", {"symbol": inst.broker_symbol, "orderListId": list_id})
            except BrokerError as e:
                if str(e.code) not in ("-2011", "-2013"):
                    raise
        for coid in singles:
            await self._cancel_protection(inst, coid)
        return len(lists) + len(singles)

    async def modify_protection(self, pos: BrokerPosition, stop: Decimal | None, target: Decimal | None) -> bool:
        """Replace the protective orders with new levels (sub-second gap between cancel and replace).

        Binance rejects a reused client order id unless the earlier order filled, so each
        replacement gets a fresh numeric suffix.
        """
        inst = self._by_canon[pos.symbol]
        spec = self._specs.get(inst.symbol) or await self.market_spec(inst)
        cid = pos.client_order_id or pos.broker_position_id
        await self._cancel_all_protection(inst, cid)
        new_stop = stop if stop is not None else pos.stop
        new_target = target if target is not None else pos.target
        version = int(time.time() * 10) % 10_000_000 or 1
        try:
            pos.protection_id = await self._place_protection(inst, cid, pos.size, new_stop, new_target, spec, version)
        except BrokerError:
            await self._market_sell(inst, pos.size, leg_id(cid, "x"))
            raise
        pos.stop, pos.target = new_stop, new_target
        return True

    async def close_position(self, pos: BrokerPosition) -> bool:
        inst = self._by_canon[pos.symbol]
        cid = pos.client_order_id or pos.broker_position_id
        await self._cancel_all_protection(inst, cid)
        order = await self._market_sell(inst, pos.size, leg_id(cid, "x"))
        return order.get("status") == "FILLED"

    async def exit_info(self, pos: BrokerPosition) -> ExitInfo:
        """The protective orders are gone: did one fill (closed) or were they all cancelled (unprotected)?

        Reads the exchange's own order history for every order carrying this position's
        client id, so it is right even if the journal holds an older protection id.
        """
        cid = pos.client_order_id or pos.broker_position_id
        inst = self._by_canon[pos.symbol]
        params = {"symbol": inst.broker_symbol, "limit": 1000}
        if pos.opened_at is not None:
            params["startTime"] = int(pos.opened_at.timestamp() * 1000) - 60_000
        orders = await self._signed("GET", "/api/v3/allOrders", params)
        mine = [o for o in orders if str(o.get("clientOrderId", "")).startswith(cid + "-")]
        for o in sorted(mine, key=lambda o: o.get("updateTime", 0), reverse=True):
            if o.get("status") != "FILLED":
                continue
            qty = _d(o.get("executedQty"))
            px = _d(o.get("cummulativeQuoteQty")) / qty if qty else None
            coid = o["clientOrderId"]
            reason = "stop loss" if "-sl" in coid else "take profit" if "-tp" in coid else "sold by engine"
            return ExitInfo("closed", px, reason)
        legs = [o for o in mine if _LEG_RE.match(o["clientOrderId"])]
        if any(o.get("status") in ("NEW", "PARTIALLY_FILLED") for o in legs):
            return ExitInfo("unknown", None, "protective order still working")
        if legs and all(o.get("status") in ("CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH") for o in legs):
            return ExitInfo("unprotected", None, "protective orders cancelled without filling")
        return ExitInfo("unknown", None, f"no protective orders found for {cid}")

    # ------------------------------------------------------------------ streaming (live public data)
    async def stream(self, instruments: list[Instrument]) -> AsyncIterator[Quote | Bar]:
        syms = [i for i in instruments if i.venue is Venue.BINANCE]
        streams = "/".join(f"{i.broker_symbol.lower()}@bookTicker/{i.broker_symbol.lower()}@kline_1m" for i in syms)
        url = f"{self._stream_url}?streams={streams}"
        backoff = Backoff()
        self.stream_connects = 0
        while True:
            try:
                async with ws_connect(url, open_timeout=15) as ws:
                    self.stream_connects += 1
                    backoff.reset()
                    async for raw in ws:
                        msg = json.loads(raw)
                        data, stream = msg.get("data", {}), msg.get("stream", "")
                        inst = self._by_symbol.get(str(data.get("s", "")).upper())
                        if inst is None:
                            continue
                        if stream.endswith("@bookTicker"):
                            yield Quote(inst.symbol, _d(data["b"]), _d(data["a"]), datetime.now(UTC))
                        elif "@kline" in stream and data.get("k", {}).get("x"):
                            k = data["k"]
                            yield Bar(inst.symbol, "M1", datetime.fromtimestamp(k["t"] / 1000, UTC), _d(k["o"]), _d(k["h"]), _d(k["l"]), _d(k["c"]), _d(k["v"]))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                delay = backoff.next()
                log.warning("Binance stream disconnected (%s); reconnecting in %.0fs", type(e).__name__, delay)
                await asyncio.sleep(delay)


RESTRICTED_LOCATION = (
    "Binance refuses connections from this computer's country (HTTP 451, restricted location). "
    "Binance blocks the United States, including US cloud servers: run the engine from an allowed "
    "country (the server lives in Melbourne for this reason)."
)


def _check(r: httpx.Response, path: str):
    if r.status_code == 451:
        raise BrokerError(f"Binance {path}: {RESTRICTED_LOCATION}", status=451)
    if r.status_code >= 500:
        raise BrokerError(f"Binance {path} server error {r.status_code}", status=r.status_code, retryable=True)
    if r.status_code >= 400:
        try:
            body = r.json()
            code, msg = body.get("code"), body.get("msg")
        except Exception:
            code, msg = None, r.text[:200]
        raise BrokerError(f"Binance {path} failed ({r.status_code}, {code}): {msg}", status=r.status_code, code=code)
    return r.json()
