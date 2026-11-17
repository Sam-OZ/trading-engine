"""Round 4 Track A: trader selection, the frozen list, the hourly consensus record and its shadow trades
(never sent), the drawdown stop, the daily-summary line and the one-look score."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest
import respx
from conftest import FakeAdapter
from sqlalchemy import select

from tradeengine import round4_follow as r4f
from tradeengine.config import InfraConfig
from tradeengine.db import (
    FollowSnapshotRow,
    OrderRow,
    ShadowEquityRow,
    ShadowTradeRow,
    StateRow,
)
from tradeengine.engine import Engine
from tradeengine.models import Venue

T0 = datetime(2026, 11, 18, 1, 2, tzinfo=UTC)
DAY_MS = 86_400_000


@pytest.fixture
async def eng(tmp_path):
    cfg = InfraConfig(_env_file=None, database_url=f"sqlite+aiosqlite:///{tmp_path / 'f.db'}", data_dir=tmp_path,
                      kill_switch_file=tmp_path / "KILL")
    e = Engine(cfg, adapters={Venue.BINANCE: FakeAdapter(Venue.BINANCE, "USDT")})
    await e.start()
    yield e
    await e.stop()


class FakeHL:
    def __init__(self, book=None, leaderboard=None, fills=None, recent=None, first_seen=None, pnl=None):
        self.book = book or {}
        self.rows = leaderboard or []
        self.fills = fills or {}
        self.recent = recent or {}
        self.first_seen = first_seen or {}
        self.pnl = pnl or {}

    async def positions(self, address):
        return self.book.get(address, (0.0, {}))

    async def leaderboard(self):
        return self.rows

    async def info(self, body):
        a = body["user"]
        if body["type"] == "portfolio":
            return [["allTime", {"accountValueHistory": [[self.first_seen.get(a, 0), "1"]], "pnlHistory": self.pnl.get(a, [])}]]
        if body["type"] == "userFillsByTime":
            if body["startTime"] > 0:
                return self.recent.get(a, [])
            return self.fills.get(a, [])
        raise AssertionError(body)


class FakeMarket:
    def __init__(self, prices=None, atr=100.0, klines=None, funding=None):
        self.prices = prices or {"BTCUSDT": 60000.0, "ETHUSDT": 3000.0}
        self._atr = atr
        self._klines = klines or {}
        self._funding = funding or []

    async def liquid(self, min_quote_volume=0):
        return {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "1000PEPE": "1000PEPEUSDT"}

    async def price(self, symbol):
        return self.prices[symbol]

    async def atr(self, symbol, days=20):
        return self._atr

    async def klines(self, symbol, interval, start_ms=None, limit=500):
        return [k for k in self._klines.get(symbol, []) if start_ms is None or k[0] >= start_ms]

    async def funding(self, symbol, start, end):
        return list(self._funding)


def row(addr, acct, pnl_all, pnl_month, name=None):
    return {"ethAddress": addr, "accountValue": str(acct), "displayName": name,
            "windowPerformances": [["month", {"pnl": str(pnl_month), "roi": "0", "vlm": "0"}],
                                   ["allTime", {"pnl": str(pnl_all), "roi": "0", "vlm": "0"}]]}


NOW_MS = int(datetime(2026, 11, 17, tzinfo=UTC).timestamp() * 1000)


def fills(n, days, coin="BTC", pnl_each=1000.0, ends_days_ago=1):
    step = days * DAY_MS / max(n - 1, 1)
    start_ms = NOW_MS - (days + ends_days_ago) * DAY_MS
    return [{"coin": coin, "px": "100", "sz": "1", "time": int(start_ms + i * step), "closedPnl": str(pnl_each if i % 2 else 0)}
            for i in range(n)]


def selection(addresses, coins=("BTC", "ETH")):
    return {"selected_at": T0.isoformat(), "scanned": len(addresses), "rejected": {},
            "traders": [{"address": a, "name": None} for a in addresses], "coins": list(coins), "rules": r4f.rules_dict()}


# ------------------------------------------------------------------ pure rules
def test_coin_mapping_and_consensus_counts_only_meaningful_positions_in_listed_coins():
    assert r4f.binance_base("BTC") == "BTC" and r4f.binance_base("kPEPE") == "1000PEPE"
    assert r4f.binance_base("xyz:NVDA") is None and r4f.binance_base("@107") is None
    book = {
        "a": (100_000, {"BTC": {"szi": 1, "value": 5_000}, "xyz:NVDA": {"szi": 1, "value": 50_000}}),
        "b": (100_000, {"BTC": {"szi": 2, "value": 9_000}, "kPEPE": {"szi": -5, "value": 3_000}}),
        "c": (100_000, {"BTC": {"szi": 1, "value": 1_000}}),  # 1% of the account: dust, not a vote
        "d": (50_000, {"ETH": {"szi": -1, "value": 4_000}, "DOGE": {"szi": 1, "value": 9_000}}),  # DOGE not listed here
    }
    assert r4f.consensus(book, {"BTC", "ETH", "1000PEPE"}) == {"BTC": (2, 0), "1000PEPE": (0, 1), "ETH": (0, 1)}
    assert r4f.wants(3, 0) == "BUY" and r4f.wants(0, 3) == "SELL" and r4f.wants(3, 1) is None and r4f.wants(2, 0) is None
    assert r4f.still_on("BUY", 2, 0) and not r4f.still_on("BUY", 1, 0) and not r4f.still_on("BUY", 4, 1)
    assert r4f.still_on("SELL", 0, 2) and not r4f.still_on("SELL", 1, 3)


def window(gain, days=400):
    """P&L history: flat until `days` ago, then `gain` by now."""
    return [[NOW_MS - days * DAY_MS, "100000"], [NOW_MS - 1, str(100000 + gain)]]


async def test_selection_skips_bots_new_idle_losing_and_hedged_accounts_and_ranks_by_realised_pnl():
    now = datetime(2026, 11, 17, tzinfo=UTC)
    old = int((now - timedelta(days=400)).timestamp() * 1000)
    hl = FakeHL(
        leaderboard=[row("bot", 5e6, 9e8, 1e6), row("good1", 2e6, 3e6, 1e5, "Ann"), row("good2", 9e5, 2e6, 5e4),
                     row("new", 1e6, 1e6, 1e4), row("hedged", 3e6, 5e8, 1e6), row("tiny", 5e4, 1e6, 1e4),
                     row("loser", 1e6, 1e6, -5), row("idle", 1e6, 1.5e6, 1e4), row("red", 1e6, 1.4e6, 1e4)],
        recent={"bot": [{}] * 2000},
        fills={"good1": fills(400, 300, pnl_each=8000), "good2": fills(300, 200, coin="kPEPE", pnl_each=9000),
               "new": fills(300, 150), "hedged": fills(300, 200, pnl_each=10),
               "idle": fills(300, 200, ends_days_ago=100), "red": fills(300, 200)},
        first_seen={a: old for a in ("good1", "good2", "hedged", "idle", "red")} | {"new": int((now - timedelta(days=60)).timestamp() * 1000)},
        pnl={"good1": window(2.5e6), "good2": window(2.0e6), "hedged": window(4e8), "idle": window(1e6), "red": window(-2e5)},
    )
    sel = await r4f.select_traders(hl, FakeMarket(), now=now, progress=lambda m: None)
    assert [t["address"] for t in sel["traders"]] == ["good1", "good2"]  # most realised trading profit first
    g1, g2 = sel["traders"]
    assert g1["realised_pnl"] == 1_600_000 and g2["realised_pnl"] == 1_350_000
    assert g1["coverage"] == 1.0 and g2["coverage"] == 1.0  # kPEPE trades as 1000PEPE on Binance
    assert g1["realised_share"] == pytest.approx(1.6e6 / 2.5e6) and g1["window_pnl"] == pytest.approx(2.5e6)
    assert sel["rejected"] == {"bot or market maker": 1, "account too new": 1, "profit mostly not from trading": 1,
                               "inactive": 1, "lost money over its kept history": 1}
    assert sel["scanned"] == 7 and sel["rules"] == r4f.rules_dict()  # "tiny" and "loser" never even scanned


# ------------------------------------------------------------------ freeze
async def test_freeze_once_and_only_under_the_same_rules(eng):
    sel = json.loads(json.dumps(selection(["a", "b", "c"])))  # as read back from the JSON file
    with pytest.raises(ValueError, match="different rules"):
        await r4f.freeze(eng.db, {**sel, "rules": {**sel["rules"], "CONSENSUS_IN": 2}})
    v = await r4f.freeze(eng.db, sel)
    assert v["frozen_at"] and len(v["traders"]) == 3
    with pytest.raises(RuntimeError, match="already frozen"):
        await r4f.freeze(eng.db, sel)


# ------------------------------------------------------------------ hourly record and shadow trades
def hour_candle(t, o, h, lo, c):
    return [int(t.timestamp() * 1000), str(o), str(h), str(lo), str(c)]


async def test_consensus_opens_a_shadow_trade_the_stop_closes_it_and_nothing_is_ever_sent(eng):
    await r4f.freeze(eng.db, selection(["a", "b", "c", "d"]))
    long_btc = {"BTC": {"szi": 1, "value": 10_000}}
    hl = FakeHL(book={"a": (100_000, long_btc), "b": (100_000, long_btc), "c": (100_000, long_btc), "d": (100_000, {})})
    m = FakeMarket(atr=1000.0, funding=[0.0001, 0.0001])
    clock = [T0]
    w = r4f.FollowWatcher(eng, hl, m, clock=lambda: clock[0])
    out = await w.run_once()
    assert out["votes"] == {"BTC": (3, 0)} and out["opened"] == ["BUY BTC"] and out["closed"] == []
    async with eng.db.sessions() as s:
        t = (await s.execute(select(ShadowTradeRow))).scalar_one()
        snaps = (await s.execute(select(FollowSnapshotRow))).scalars().all()
    assert len(snaps) == 4
    entry = 60000 * 1.0002
    assert float(t.entry) == pytest.approx(entry) and float(t.stop) == pytest.approx(entry - 3000)
    assert float(t.risk_usd) == pytest.approx(25.0) and float(t.size) == pytest.approx(25 / 3000)
    assert "3 long / 0 short of 4" in t.reason

    # two hours later: the 02:00 candle dipped through the stop (the open was above it)
    m._klines["BTCUSDT"] = [hour_candle(datetime(2026, 11, 18, 2, tzinfo=UTC), 60000, 60100, 56000, 57000)]
    clock[0] = T0 + timedelta(hours=2)
    out = await w.run_once()
    assert out["closed"] == ["BTC stop"] and out["opened"] == []  # consensus still on, but after a stop the coin waits
    async with eng.db.sessions() as s:
        t = (await s.execute(select(ShadowTradeRow).where(ShadowTradeRow.id == t.id))).scalar_one()
    exit_px = (entry - 3000) * (1 - 0.0002)
    size = 25 / 3000
    fees = 0.0005 * (entry * size + exit_px * size)
    funding = 0.0002 * entry * size
    assert t.exit_reason == "stop" and float(t.exit) == pytest.approx(exit_px)
    assert float(t.pnl) == pytest.approx((exit_px - entry) * size - fees - funding, rel=1e-6)
    assert float(t.r) == pytest.approx(float(t.pnl) / 25, rel=1e-6) and float(t.r) < -1  # a full loss plus costs
    async with eng.db.sessions() as s:
        assert (await s.execute(select(OrderRow))).first() is None  # shadow only: no order ever reached the journal

    # the consensus has to break and form again before BTC is bought again
    clock[0] = T0 + timedelta(hours=3)
    assert (await w.run_once())["opened"] == []
    hl.book["b"] = (100_000, {})
    clock[0] = T0 + timedelta(hours=4)
    assert (await w.run_once())["opened"] == []  # 2 long: broken
    hl.book["b"] = (100_000, long_btc)
    clock[0] = T0 + timedelta(hours=5)
    assert (await w.run_once())["opened"] == ["BUY BTC"]  # formed again


async def test_a_short_consensus_and_its_break_close_at_the_market(eng):
    await r4f.freeze(eng.db, selection(["a", "b", "c"]))
    short = {"ETH": {"szi": -3, "value": 9_000}}
    hl = FakeHL(book={k: (100_000, short) for k in "abc"})
    m = FakeMarket(atr=50.0)
    clock = [T0]
    w = r4f.FollowWatcher(eng, hl, m, clock=lambda: clock[0])
    assert (await w.run_once())["opened"] == ["SELL ETH"]
    hl.book["b"] = (100_000, {"ETH": {"szi": 2, "value": 8_000}})  # one trader flips long: the consensus is gone
    m.prices["ETHUSDT"] = 2900.0
    clock[0] = T0 + timedelta(minutes=30)
    out = await w.run_once()
    assert out["closed"] == ["ETH consensus"]
    async with eng.db.sessions() as s:
        t = (await s.execute(select(ShadowTradeRow))).scalar_one()
    assert t.exit_reason.startswith("consensus broke (1 long, 2 short)") and float(t.pnl) > 0  # short, price fell


async def test_ten_percent_shadow_drawdown_ends_new_entries(eng):
    await r4f.freeze(eng.db, selection(["a", "b", "c"]))
    async with eng.db.sessions() as s, s.begin():
        s.add(ShadowTradeRow(coin="ETH", symbol="ETHUSDT", side="BUY", opened_at=T0 - timedelta(days=5), entry=Decimal("1"),
                             stop=Decimal("0.5"), size=Decimal("1"), risk_usd=Decimal("25"), reason="x",
                             closed_at=T0 - timedelta(days=4), exit=Decimal("1"), exit_reason="stop", pnl=Decimal("-600"), r=-24))
    long_btc = {"BTC": {"szi": 1, "value": 10_000}}
    w = r4f.FollowWatcher(eng, FakeHL(book={k: (100_000, long_btc) for k in "abc"}), FakeMarket(), clock=lambda: T0)
    assert (await w.run_once())["opened"] == []
    async with eng.db.sessions() as s:
        assert await s.get(StateRow, r4f.KILLED_KEY) is not None
    st = await r4f.stats(eng.db, T0)
    assert st.killed and "STOPPED at 10% drawdown" in r4f.summary_line(st)


async def test_idle_until_frozen_and_when_switched_off(eng):
    w = r4f.FollowWatcher(eng, FakeHL(), FakeMarket(), clock=lambda: T0)
    assert await w.run_once() is None
    await r4f.freeze(eng.db, selection(["a", "b", "c"]))
    await eng.settings_store.update({"round4_follow_enabled": False}, "t")
    assert await w.run_once() is None


# ------------------------------------------------------------------ summary line and the one-look score
async def test_daily_summary_carries_the_track_a_line(eng):
    from tradeengine.notify import summary_text

    assert "Track A" not in await summary_text(eng, T0)
    await r4f.freeze(eng.db, selection(["a", "b", "c"]))
    assert "Track A shadow (since " in await summary_text(eng, T0)


async def test_score_waits_for_90_days_and_40_trades_then_locks(eng):
    await r4f.freeze(eng.db, selection(["a", "b", "c"]))
    rs = [1.5, -1.0, 2.0, -1.0, 0.8] * 8  # 40 trades, profit factor 1.72
    async with eng.db.sessions() as s, s.begin():
        for i, r in enumerate(rs):
            s.add(ShadowTradeRow(coin="BTC", symbol="BTCUSDT", side="BUY", opened_at=T0, entry=Decimal("1"), stop=Decimal("0.9"),
                                 size=Decimal("1"), risk_usd=Decimal("25"), reason="x", closed_at=T0 + timedelta(days=i),
                                 exit=Decimal("1"), exit_reason="x", pnl=Decimal(str(25 * r)), r=r))
        for i in range(10):
            s.add(ShadowEquityRow(ts=T0 + timedelta(days=i), equity=Decimal(str(5000 + 20 * i - (40 if i == 5 else 0)))))
    with pytest.raises(RuntimeError, match="Too early"):
        await r4f.score(eng.db, datetime.now(UTC) + timedelta(days=30))
    ok, why, st = await r4f.score(eng.db, datetime.now(UTC) + timedelta(days=91))
    assert ok and why == [] and st.closed == 40 and st.profit_factor == pytest.approx(34.4 / 16.0)
    with pytest.raises(RuntimeError, match="already scored"):
        await r4f.score(eng.db, datetime.now(UTC) + timedelta(days=92))


# ------------------------------------------------------------------ an unreliable connection must not skew the selection
@respx.mock
async def test_hyperliquid_calls_retry_hiccups_and_fail_fast_on_real_errors():
    waits = []

    async def no_wait(s):
        waits.append(s)

    route = respx.post(r4f.HL_INFO).mock(side_effect=[httpx.Response(403), httpx.Response(502), httpx.ConnectError("x"),
                                                      httpx.Response(200, text="<html>"), httpx.Response(200, json=[1, 2])])
    async with httpx.AsyncClient() as client:
        hl = r4f.Hyperliquid(client, sleep=no_wait)
        assert await hl.info({"type": "portfolio", "user": "a"}) == [1, 2]
        assert route.call_count == 5 and waits[:4] == [15, 30, 60, 120]  # then the normal pacing pause
        route.side_effect = [httpx.Response(422, json={"error": "bad user"})]
        with pytest.raises(httpx.HTTPStatusError):  # a real error is not retried
            await hl.info({"type": "portfolio", "user": "b"})
        route.side_effect = [httpx.Response(503)] * 6
        with pytest.raises(httpx.HTTPError, match="kept failing \\(HTTP 503\\) for portfolio"):
            await hl.info({"type": "portfolio", "user": "c"})


async def test_unread_accounts_get_a_second_pass_and_an_incomplete_scan_cannot_be_frozen(eng):
    now = datetime(2026, 11, 17, tzinfo=UTC)
    old = int((now - timedelta(days=400)).timestamp() * 1000)

    class Flaky(FakeHL):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.fails = {"blip": 1, "dead": 99}

        async def sleep(self, s):
            pass

        async def info(self, body):
            if self.fails.get(body["user"], 0) > 0:
                self.fails[body["user"]] -= 1
                raise httpx.HTTPError("Hyperliquid kept failing (HTTP 403) for userFillsByTime")
            return await super().info(body)

    good = {"fills": fills(400, 300, pnl_each=8000), "pnl": window(2.5e6)}
    hl = Flaky(leaderboard=[row(a, 2e6, 3e6, 1e5) for a in ("ok", "blip", "dead")],
               fills={a: good["fills"] for a in ("ok", "blip", "dead")}, pnl={a: good["pnl"] for a in ("ok", "blip", "dead")},
               first_seen={a: old for a in ("ok", "blip", "dead")})
    sel = await r4f.select_traders(hl, FakeMarket(), now=now, progress=lambda m: None)
    assert sorted(t["address"] for t in sel["traders"]) == ["blip", "ok"]  # "blip" recovered on the second pass
    assert sel["unread"] == 1 and sel["unread_addresses"] == ["dead"]
    assert sel["unread_errors"] == {"HTTPError: Hyperliquid kept failing (HTTP 403)": 1}
    with pytest.raises(ValueError, match="could not read 1 of 3 accounts"):  # 33% unread: not a fair ranking
        await r4f.freeze(eng.db, json.loads(json.dumps(sel)))
    assert await r4f.load_selection(eng.db) is None
