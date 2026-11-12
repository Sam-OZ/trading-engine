"""Binance Spot Demo Mode adapter against mocked HTTP and a local WebSocket server."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime
from decimal import Decimal as D
from urllib.parse import parse_qs, urlsplit

import httpx
import respx
from conftest import BTCUSDT
from websockets.asyncio.server import serve

from tradeengine.brokers.binance import (
    DEMO_TRADE_REST,
    LIVE_PUBLIC_REST,
    BinanceSpotDemoAdapter,
)
from tradeengine.ids import client_order_id
from tradeengine.models import (
    DEFAULT_INSTRUMENTS,
    BrokerPosition,
    OrderPlan,
    Side,
    Source,
    TradeIntent,
    Venue,
)

INSTS = list(DEFAULT_INSTRUMENTS.values())
T = datetime(2026, 11, 11, 14, tzinfo=UTC)
CID = client_order_id(Source.STRATEGY, "t", "BTC/USDT", Side.BUY, T)

EXCHANGE_INFO = {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "filters": [
    {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
    {"filterType": "LOT_SIZE", "minQty": "0.00001000", "stepSize": "0.00001000"},
    {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
]}]}


def adapter(**kw) -> BinanceSpotDemoAdapter:
    return BinanceSpotDemoAdapter(
        "APIKEY", "SECRET", INSTS,
        trade_client=httpx.AsyncClient(base_url=DEMO_TRADE_REST),
        market_client=httpx.AsyncClient(base_url=LIVE_PUBLIC_REST),
        **kw,
    )


def q(call) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(str(call.request.url)).query).items()}


def plan(target=D("61200")) -> OrderPlan:
    i = TradeIntent("BTC/USDT", Side.BUY, D("60000"), D("59400"), target, Source.STRATEGY, "t", T)
    return OrderPlan(CID, BTCUSDT, i, D("0.00420"), D("2.5"))


def common_routes(mock):
    mock.get(f"{DEMO_TRADE_REST}/api/v3/exchangeInfo").mock(return_value=httpx.Response(200, json=EXCHANGE_INFO))
    mock.get(f"{LIVE_PUBLIC_REST}/api/v3/ticker/bookTicker").mock(
        return_value=httpx.Response(200, json={"symbol": "BTCUSDT", "bidPrice": "60010.00", "askPrice": "60010.01"})
    )


FILLED = {
    "orderId": 555, "status": "FILLED", "executedQty": "0.00420000", "cummulativeQuoteQty": "252.04200000",
    "fills": [{"price": "60010.00", "qty": "0.00420000", "commission": "0.00000420", "commissionAsset": "BTC"}],
}


def test_signature_matches_binance_documentation_example():
    a = BinanceSpotDemoAdapter("k", "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j", INSTS)
    params = {"symbol": "LTCBTC", "side": "BUY", "type": "LIMIT", "timeInForce": "GTC", "quantity": "1", "price": "0.1", "recvWindow": 5000, "timestamp": 1499827319559}
    assert a.sign(params).endswith("signature=c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71")


@respx.mock
async def test_buy_then_oco_with_fee_adjusted_quantity():
    common_routes(respx)
    order = respx.post(f"{DEMO_TRADE_REST}/api/v3/order").mock(return_value=httpx.Response(200, json=FILLED))
    oco = respx.post(f"{DEMO_TRADE_REST}/api/v3/orderList/oco").mock(return_value=httpx.Response(200, json={"orderListId": 9}))
    res = await adapter().open_position(plan())
    o = q(order.calls[0])
    assert o["type"] == "MARKET" and o["side"] == "BUY" and o["quantity"] == "0.0042" and o["newClientOrderId"] == CID
    assert "signature" in o and order.calls[0].request.headers["X-MBX-APIKEY"] == "APIKEY"
    p = q(oco.calls[0])
    assert p["side"] == "SELL" and p["quantity"] == "0.00419"  # 0.0042 - 0.0000042 fee, rounded down
    assert p["aboveType"] == "LIMIT_MAKER" and p["abovePrice"] == "61200"
    assert p["belowType"] == "STOP_LOSS" and p["belowStopPrice"] == "59400"
    assert p["listClientOrderId"] == f"{CID}-oco" and p["belowClientOrderId"] == f"{CID}-sl"
    assert res.ok and res.protected and res.protection_id == f"{CID}-oco" and res.fill_price == D("60010")
    # Keys never go to the live host.
    live_calls = [c for c in respx.calls if c.request.url.host == "api.binance.com"]
    assert live_calls and all("X-MBX-APIKEY" not in c.request.headers for c in live_calls)


@respx.mock
async def test_stop_only_when_no_target():
    common_routes(respx)
    order = respx.post(f"{DEMO_TRADE_REST}/api/v3/order").mock(side_effect=[httpx.Response(200, json=FILLED), httpx.Response(200, json={"status": "NEW"})])
    res = await adapter().open_position(plan(target=None))
    stop = q(order.calls[1])
    assert stop["type"] == "STOP_LOSS" and stop["stopPrice"] == "59400" and stop["newClientOrderId"] == f"{CID}-sl"
    assert res.ok and res.protection_id == f"{CID}-sl"


@respx.mock
async def test_protection_failure_sells_immediately():
    common_routes(respx)
    order = respx.post(f"{DEMO_TRADE_REST}/api/v3/order").mock(side_effect=[httpx.Response(200, json=FILLED), httpx.Response(200, json={"status": "FILLED"})])
    respx.post(f"{DEMO_TRADE_REST}/api/v3/orderList/oco").mock(return_value=httpx.Response(400, json={"code": -1013, "msg": "Filter failure"}))
    res = await adapter().open_position(plan())
    sell = q(order.calls[1])
    assert sell["side"] == "SELL" and sell["type"] == "MARKET" and sell["newClientOrderId"] == f"{CID}-x"
    assert not res.ok and "sold" in res.error


async def test_sell_to_open_refused():
    i = TradeIntent("BTC/USDT", Side.SELL, D("60000"), D("60600"), None, Source.STRATEGY, "t", T)
    res = await adapter().open_position(OrderPlan(CID, BTCUSDT, i, D("0.001"), D("1")))
    assert not res.ok and "long-only" in res.error


OPEN_ORDERS = [
    {"symbol": "BTCUSDT", "orderId": 1, "orderListId": 9, "clientOrderId": f"{CID}-sl", "origQty": "0.00419", "executedQty": "0", "stopPrice": "59400.00", "price": "0", "type": "STOP_LOSS"},
    {"symbol": "BTCUSDT", "orderId": 2, "orderListId": 9, "clientOrderId": f"{CID}-tp", "origQty": "0.00419", "executedQty": "0", "price": "61200.00", "type": "LIMIT_MAKER"},
    {"symbol": "BTCUSDT", "orderId": 3, "orderListId": -1, "clientOrderId": "manual-order", "origQty": "1", "executedQty": "0", "price": "1"},
]


@respx.mock
async def test_list_positions_groups_our_protective_orders():
    respx.get(f"{DEMO_TRADE_REST}/api/v3/openOrders").mock(
        side_effect=lambda req: httpx.Response(200, json=OPEN_ORDERS if "BTCUSDT" in str(req.url) else [])
    )
    (p,) = await adapter().list_positions()
    assert p.client_order_id == CID and p.size == D("0.00419") and p.stop == D("59400") and p.target == D("61200")
    assert p.protection_id == f"{CID}-oco"


def open_orders_route(mock, orders):
    return mock.get(f"{DEMO_TRADE_REST}/api/v3/openOrders").mock(
        side_effect=lambda req: httpx.Response(200, json=orders if "BTCUSDT" in str(req.url) else [])
    )


# The live OCO is version 7 while the journal still remembers the original "-oco":
# modify/close must act on what is live at the exchange, not the stored id.
LIVE_V7 = [
    {"symbol": "BTCUSDT", "orderId": 11, "orderListId": 42, "clientOrderId": f"{CID}-sl7", "origQty": "0.00419", "stopPrice": "59500"},
    {"symbol": "BTCUSDT", "orderId": 12, "orderListId": 42, "clientOrderId": f"{CID}-tp7", "origQty": "0.00419", "price": "61200"},
    {"symbol": "BTCUSDT", "orderId": 13, "orderListId": 43, "clientOrderId": "te-ffffffffffffffffffff-sl", "origQty": "1", "stopPrice": "1"},
]


@respx.mock
async def test_modify_cancels_live_oco_and_places_fresh_ids():
    common_routes(respx)
    open_orders_route(respx, LIVE_V7)
    cancel = respx.delete(f"{DEMO_TRADE_REST}/api/v3/orderList").mock(return_value=httpx.Response(200, json={}))
    oco = respx.post(f"{DEMO_TRADE_REST}/api/v3/orderList/oco").mock(return_value=httpx.Response(200, json={}))
    bp = BrokerPosition(Venue.BINANCE, "BTC/USDT", Side.BUY, D("0.00419"), D("60000"), D("59400"), D("61200"), CID, f"{CID}-oco", CID)
    assert await adapter().modify_protection(bp, D("59800"), None)
    assert [q(c)["orderListId"] for c in cancel.calls] == ["42"]  # only ours, the live one
    p = q(oco.calls[0])
    assert re.fullmatch(rf"{CID}-oco\d+", p["listClientOrderId"]) and p["listClientOrderId"] != f"{CID}-oco"
    assert len(p["listClientOrderId"]) <= 36 and p["belowStopPrice"] == "59800" and p["abovePrice"] == "61200"
    assert bp.protection_id == p["listClientOrderId"] and bp.stop == D("59800")


@respx.mock
async def test_close_cancels_live_protection_then_sells():
    open_orders_route(respx, LIVE_V7)
    cancel = respx.delete(f"{DEMO_TRADE_REST}/api/v3/orderList").mock(return_value=httpx.Response(200, json={}))
    sell = respx.post(f"{DEMO_TRADE_REST}/api/v3/order").mock(return_value=httpx.Response(200, json={"status": "FILLED"}))
    bp = BrokerPosition(Venue.BINANCE, "BTC/USDT", Side.BUY, D("0.00419"), D("60000"), D("59400"), D("61200"), "555", f"{CID}-oco", CID)
    assert await adapter().close_position(bp)
    assert [q(c)["orderListId"] for c in cancel.calls] == ["42"]
    s = q(sell.calls[0])
    assert s["quantity"] == "0.00419" and s["side"] == "SELL" and s["newClientOrderId"] == f"{CID}-x"


def all_orders_route(mock, orders):
    return mock.get(f"{DEMO_TRADE_REST}/api/v3/allOrders").mock(return_value=httpx.Response(200, json=orders))


@respx.mock
async def test_exit_info_finds_fill_on_newer_oco_version():
    all_orders_route(respx, [
        {"clientOrderId": CID, "status": "FILLED", "executedQty": "0.0042", "cummulativeQuoteQty": "252", "updateTime": 1},
        {"clientOrderId": f"{CID}-sl", "status": "CANCELED", "updateTime": 2},
        {"clientOrderId": f"{CID}-tp", "status": "CANCELED", "updateTime": 2},
        {"clientOrderId": f"{CID}-sl7", "status": "FILLED", "executedQty": "0.00419", "cummulativeQuoteQty": "249.305", "updateTime": 3},
        {"clientOrderId": f"{CID}-tp7", "status": "EXPIRED", "updateTime": 3},
        {"clientOrderId": "someone-else", "status": "FILLED", "executedQty": "1", "cummulativeQuoteQty": "1", "updateTime": 4},
    ])
    bp = BrokerPosition(Venue.BINANCE, "BTC/USDT", Side.BUY, D("0.00419"), D("60000"), D("59400"), D("61200"), "555", f"{CID}-oco", CID)
    info = await adapter().exit_info(bp)
    assert info.status == "closed" and info.reason == "stop loss" and info.exit_price == D("59500")


@respx.mock
async def test_exit_info_detects_cancelled_protection():
    all_orders_route(respx, [
        {"clientOrderId": CID, "status": "FILLED", "executedQty": "0.0042", "cummulativeQuoteQty": "252", "updateTime": 1},
        {"clientOrderId": f"{CID}-sl", "status": "CANCELED", "updateTime": 2},
        {"clientOrderId": f"{CID}-tp", "status": "CANCELED", "updateTime": 2},
    ])
    bp = BrokerPosition(Venue.BINANCE, "BTC/USDT", Side.BUY, D("0.00419"), D("60000"), D("59400"), D("61200"), "555", f"{CID}-oco", CID)
    assert (await adapter().exit_info(bp)).status == "unprotected"


async def test_stream_quotes_and_closed_klines_with_reconnect():
    async def handler(ws):
        await ws.send(json.dumps({"stream": "btcusdt@bookTicker", "data": {"s": "BTCUSDT", "b": "60000.00", "a": "60000.02"}}))
        await ws.send(json.dumps({"stream": "btcusdt@kline_1m", "data": {"s": "BTCUSDT", "k": {"t": 1762869600000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "x": False}}}))
        await ws.send(json.dumps({"stream": "btcusdt@kline_1m", "data": {"s": "BTCUSDT", "k": {"t": 1762869600000, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10", "x": True}}}))
        await ws.close()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        a = adapter(stream_url=f"ws://127.0.0.1:{port}/stream")
        events = []

        async def consume():
            async for ev in a.stream([BTCUSDT]):
                events.append(ev)
                if len(events) == 3:
                    return

        await asyncio.wait_for(consume(), timeout=10)
    assert events[0].mid == D("60000.01")
    assert events[1].timeframe == "M1" and events[1].close == D("1.5")  # only the closed kline
    assert a.stream_connects == 2
