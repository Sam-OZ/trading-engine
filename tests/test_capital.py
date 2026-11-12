"""Capital.com adapter against mocked HTTP and a local WebSocket server."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import httpx
import pytest
import respx
from conftest import EURUSD
from websockets.asyncio.server import serve

from tradeengine.brokers.capital import DEMO_REST, CapitalComAdapter
from tradeengine.ids import client_order_id
from tradeengine.models import DEFAULT_INSTRUMENTS, OrderPlan, Side, Source, TradeIntent

INSTS = list(DEFAULT_INSTRUMENTS.values())
SESSION = {"currentAccountId": "ACC1", "accountType": "CFD"}


def adapter(**kw) -> CapitalComAdapter:
    return CapitalComAdapter("KEY", "me@example.com", "pw", INSTS, client=httpx.AsyncClient(base_url=DEMO_REST), confirm_delay=0, **kw)


def login_route(mock):
    return mock.post(f"{DEMO_REST}/api/v1/session").mock(
        return_value=httpx.Response(200, json=SESSION, headers={"CST": "cst1", "X-SECURITY-TOKEN": "xst1"})
    )


def plan() -> OrderPlan:
    t = datetime(2026, 11, 11, 14, tzinfo=UTC)
    i = TradeIntent("EUR/USD", Side.BUY, D("1.10010"), D("1.097613"), D("1.10610"), Source.STRATEGY, "t", t)
    return OrderPlan(client_order_id(i.source, i.strategy, i.symbol, i.side, t), EURUSD, i, D("1000"), D("2.4"))


MARKET = {
    "instrument": {"epic": "EURUSD", "lotSize": 1, "currency": "USD"},
    "dealingRules": {"minDealSize": {"value": 100}, "minSizeIncrement": {"value": 1}},
    "snapshot": {"marketStatus": "TRADEABLE", "bid": 1.0999, "offer": 1.1001, "decimalPlacesFactor": 5},
}


@respx.mock
async def test_login_sends_key_and_reads_tokens():
    route = login_route(respx)
    a = adapter()
    await a.connect()
    req = route.calls[0].request
    assert req.headers["X-CAP-API-KEY"] == "KEY"
    assert json.loads(req.content) == {"identifier": "me@example.com", "password": "pw", "encryptedPassword": False}
    assert a._cst == "cst1" and a._xst == "xst1"


@respx.mock
async def test_account_picks_current_account_and_computes_equity():
    login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/accounts").mock(
        return_value=httpx.Response(
            200,
            json={"accounts": [
                {"accountId": "OTHER", "preferred": True, "currency": "USD", "balance": {"balance": 1, "profitLoss": 0, "available": 1}},
                {"accountId": "ACC1", "currency": "AUD", "balance": {"balance": 10000, "profitLoss": -25.5, "available": 9000}},
            ]},
        )
    )
    acct = await adapter().account()
    assert acct.currency == "AUD" and acct.equity == D("9974.5") and acct.available == D("9000")


@respx.mock
async def test_expired_session_relogs_in_once():
    login = login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/ping").mock(side_effect=[httpx.Response(401, json={"errorCode": "error.invalid.session.token"}), httpx.Response(200, json={"status": "OK"})])
    a = adapter()
    assert await a.ping()
    assert login.call_count == 2


@respx.mock
async def test_market_spec_and_history():
    login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/markets/EURUSD").mock(return_value=httpx.Response(200, json=MARKET))
    respx.get(f"{DEMO_REST}/api/v1/prices/EURUSD").mock(
        return_value=httpx.Response(
            200,
            json={"prices": [
                {"snapshotTimeUTC": "2026-11-11T13:00:00", "openPrice": {"bid": 1.1, "ask": 1.1002}, "highPrice": {"bid": 1.101, "ask": 1.1012},
                 "lowPrice": {"bid": 1.099, "ask": 1.0992}, "closePrice": {"bid": 1.1005, "ask": 1.1007}, "lastTradedVolume": 42},
            ]},
        )
    )
    a = adapter()
    spec = await a.market_spec(EURUSD)
    assert spec.min_size == D("100") and spec.price_tick == D("0.00001") and spec.tradeable
    bars = await a.history(EURUSD, "M15", datetime(2026, 11, 11, tzinfo=UTC), datetime(2026, 11, 12, tzinfo=UTC))
    assert len(bars) == 1 and bars[0].open == D("1.1001") and bars[0].close == D("1.1006")
    assert bars[0].open_time == datetime(2026, 11, 11, 13, tzinfo=UTC)


@respx.mock
async def test_open_position_confirms_and_verifies_stop():
    login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/markets/EURUSD").mock(return_value=httpx.Response(200, json=MARKET))
    post = respx.post(f"{DEMO_REST}/api/v1/positions").mock(return_value=httpx.Response(200, json={"dealReference": "o_ref1"}))
    respx.get(f"{DEMO_REST}/api/v1/confirms/o_ref1").mock(
        side_effect=[httpx.Response(404), httpx.Response(200, json={"dealStatus": "ACCEPTED", "level": 1.10012, "size": 1000, "affectedDeals": [{"dealId": "DEAL1", "status": "OPENED"}]})]
    )
    respx.get(f"{DEMO_REST}/api/v1/positions").mock(
        return_value=httpx.Response(200, json={"positions": [{
            "position": {"dealId": "DEAL1", "direction": "BUY", "size": 1000, "level": 1.10012, "stopLevel": 1.09761, "profitLevel": 1.1061, "createdDateUTC": "2026-11-11T14:00:01.000", "upl": -0.2},
            "market": {"epic": "EURUSD", "bid": 1.1, "offer": 1.1002}}]})
    )
    a = adapter()
    await a.market_spec(EURUSD)
    res = await a.open_position(plan())
    body = json.loads(post.calls[0].request.content)
    assert body == {"epic": "EURUSD", "direction": "BUY", "size": 1000.0, "guaranteedStop": False, "stopLevel": 1.09761, "profitLevel": 1.1061}
    assert res.ok and res.broker_position_id == "DEAL1" and res.protected and res.fill_price == D("1.10012")
    (p,) = await a.list_positions()
    assert p.symbol == "EUR/USD" and p.stop == D("1.09761") and p.unrealised_pnl == D("-0.2")


@respx.mock
async def test_rejected_and_unconfirmed_deals():
    login_route(respx)
    respx.post(f"{DEMO_REST}/api/v1/positions").mock(return_value=httpx.Response(200, json={"dealReference": "o_r"}))
    respx.get(f"{DEMO_REST}/api/v1/confirms/o_r").mock(return_value=httpx.Response(200, json={"dealStatus": "REJECTED", "reason": "MARKET_CLOSED"}))
    res = await adapter().open_position(plan())
    assert not res.ok and "MARKET_CLOSED" in res.error
    respx.get(f"{DEMO_REST}/api/v1/confirms/o_r").mock(return_value=httpx.Response(404))
    res = await adapter(confirm_attempts=2).open_position(plan())
    assert not res.ok and res.error.startswith("unconfirmed")


@respx.mock
async def test_find_order_matches_unknown_position_only():
    login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/positions").mock(
        return_value=httpx.Response(200, json={"positions": [
            {"position": {"dealId": "OLD", "direction": "BUY", "size": 1000, "level": 1.1, "stopLevel": 1.09, "createdDateUTC": "2026-11-11T13:00:00"}, "market": {"epic": "EURUSD"}},
            {"position": {"dealId": "NEW", "direction": "BUY", "size": 1000, "level": 1.1, "stopLevel": 1.09, "createdDateUTC": "2026-11-11T14:00:02"}, "market": {"epic": "EURUSD"}},
        ]})
    )
    res = await adapter().find_order(plan(), datetime(2026, 11, 11, 14, 0, tzinfo=UTC), known_ids={"OLD"})
    assert res and res.broker_position_id == "NEW" and res.protected


@respx.mock
async def test_server_errors_are_retryable():
    login_route(respx)
    respx.get(f"{DEMO_REST}/api/v1/accounts").mock(return_value=httpx.Response(503))
    from tradeengine.brokers.base import BrokerError

    with pytest.raises(BrokerError) as e:
        await adapter().account()
    assert e.value.retryable


@respx.mock
async def test_stream_parses_quotes_and_reconnects():
    login_route(respx)
    seen_subs: list[dict] = []

    async def handler(ws):
        sub = json.loads(await ws.recv())
        seen_subs.append(sub)
        await ws.send(json.dumps({"destination": "quote", "payload": {"epic": "EURUSD", "bid": 1.1, "ofr": 1.1002, "timestamp": 1762869600000}}))
        await ws.close()  # simulate a dropped connection

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        a = adapter(stream_url=f"ws://127.0.0.1:{port}")
        quotes = []

        async def consume():
            async for q in a.stream([EURUSD]):
                quotes.append(q)
                if len(quotes) == 2:
                    return

        await asyncio.wait_for(consume(), timeout=10)
    assert [q.mid for q in quotes] == [D("1.1001"), D("1.1001")]
    assert a.stream_connects == 2
    assert seen_subs[0]["destination"] == "marketData.subscribe" and seen_subs[0]["payload"] == {"epics": ["EURUSD"]}
    assert seen_subs[0]["cst"] == "cst1"


@respx.mock
async def test_history_requests_windows_of_at_most_1000_bars():
    login_route(respx)
    route = respx.get(f"{DEMO_REST}/api/v1/prices/EURUSD").mock(return_value=httpx.Response(200, json={"prices": []}))
    start = datetime(2026, 1, 1, tzinfo=UTC)
    await adapter().history(EURUSD, "M15", start, start + timedelta(days=30))  # 2,880 bars -> 3 windows
    windows = [(c.request.url.params["from"], c.request.url.params["to"]) for c in route.calls]
    assert windows == [
        ("2026-01-01T00:00:00", "2026-01-11T10:00:00"),
        ("2026-01-11T10:00:00", "2026-01-21T20:00:00"),
        ("2026-01-21T20:00:00", "2026-01-31T00:00:00"),
    ]
