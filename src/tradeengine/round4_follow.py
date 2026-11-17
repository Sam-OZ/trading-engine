"""Round 4, Track A: follow good Hyperliquid traders, scored on a FORWARD shadow record. Approved by Sam
17 Nov 2026 (plan: Claude Docs "Round 4 plan - follow good traders and test Kronos"). The constants below ARE
the pre-registration, frozen once the selection is frozen.

1. Select (once, on a desktop: the leaderboard is ~40 MB): traders on Hyperliquid's public leaderboard with
   an account >= $100k, profitable all-time and this month, >= 180 days old, at most 50 fills a day (no bots
   or market makers), >= 100 closed trades and >= 90 days in the history Hyperliquid keeps, at least half
   of all-time profit realised from trading (not funding / off-exchange hedges), and >= 70% of their volume
   in coins Binance futures lists with >= $100M a day. Best realised trading P&L first, up to 15.
2. Freeze (once, on the server): the list is stored and never changed.
3. Record (hourly, on the server, read-only): their open positions. A coin has a LONG consensus when at least
   3 followed traders are meaningfully long it (>= 2% of their account) and none is short; SHORT the mirror.
   A shadow trade opens at Binance futures' price + slippage, 0.5% risk of shadow equity, stop 3 x ATR(20)
   daily; it closes when the consensus breaks (fewer than 2 still in, or anyone on the other side) or the stop
   is touched (checked on Binance hourly candles). After a stop, that coin waits for the consensus to break and
   form again before a new entry (no hourly re-buying into the same move). Fees 0.05% a side, 2 bps slippage,
   real funding payments.
   Limits as the live engine: 4 open, 1 per coin, 5:1 leverage, 2% daily loss, 10% drawdown ends the test.
   NO ORDER IS EVER SENT.
4. Score (once): after >= 90 days and >= 40 closed shadow trades: profit factor >= 1.2, drawdown < 10%,
   5th percentile of 10,000 seeded bootstrap means of trade R > 0.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
from sqlalchemy import func, select

from .db import FollowSnapshotRow, ShadowEquityRow, ShadowTradeRow, StateRow
from .round2 import Rules, bootstrap_lower_bound
from .state import log_event

log = logging.getLogger(__name__)

HL_INFO = "https://api.hyperliquid.xyz/info"
HL_LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
FUTURES_BASES = ("https://fapi.binance.com", "https://demo-fapi.binance.com")  # public market data only

# --- selection (frozen) ---
MIN_ACCOUNT = 100_000.0
MIN_AGE_DAYS = 180
MAX_FILLS_PER_DAY = 50.0
MIN_CLOSING_FILLS = 100
MIN_SPAN_DAYS = 90.0
MIN_REALISED_SHARE = 0.5
MIN_COVERAGE = 0.7
MIN_COIN_QUOTE_VOLUME = 100e6
MAX_TRADERS = 15
MAX_SCAN = 600
# --- copy rule (frozen) ---
CONSENSUS_IN = 3
CONSENSUS_HOLD = 2
MIN_POSITION_SHARE = 0.02
RISK_PCT = 0.005
STOP_ATR = 3.0
ATR_DAYS = 20
FEE_RATE = 0.0005
SLIPPAGE_BPS = 2.0
START_EQUITY = 5000.0
MAX_OPEN = 4
MAX_LEVERAGE = 5.0
DAILY_LOSS_PCT = 2.0
MAX_DD_PCT = 10.0
# --- score (frozen) ---
MIN_RECORD_DAYS = 90
RULES = Rules(holdout_min_pf=1.2, holdout_max_dd_pct=10.0, holdout_min_trades=40, bootstrap=True)

SELECTION_KEY = "round4_follow_selection"
KILLED_KEY = "round4_follow_killed"
STOPPED_KEY = "round4_follow_stopped"  # coin -> side stopped out; cleared once that consensus breaks
SCORE_KEY = "round4_holdout:follow"


FROZEN = ("MIN_ACCOUNT", "MIN_AGE_DAYS", "MAX_FILLS_PER_DAY", "MIN_CLOSING_FILLS", "MIN_SPAN_DAYS", "MIN_REALISED_SHARE",
          "MIN_COVERAGE", "MIN_COIN_QUOTE_VOLUME", "MAX_TRADERS", "MAX_SCAN", "CONSENSUS_IN", "CONSENSUS_HOLD",
          "MIN_POSITION_SHARE", "RISK_PCT", "STOP_ATR", "ATR_DAYS", "FEE_RATE", "SLIPPAGE_BPS", "START_EQUITY", "MAX_OPEN",
          "MAX_LEVERAGE", "DAILY_LOSS_PCT", "MAX_DD_PCT", "MIN_RECORD_DAYS")


def rules_dict() -> dict:
    """Every pre-registered number, stored with the selection so a later code change cannot slip in unnoticed."""
    g = globals()
    return {**{k: g[k] for k in FROZEN}, "RULES": asdict(RULES)}


def binance_base(coin: str) -> str | None:
    """Hyperliquid coin -> Binance futures base asset. Spot (@n) and stock/commodity (xyz:...) markets: None."""
    if coin.startswith("@") or ":" in coin:
        return None
    if coin.startswith("k") and coin[1:].isupper():
        return "1000" + coin[1:]
    return coin


# ------------------------------------------------------------------ public data clients
class Hyperliquid:
    """Hyperliquid's public info API, paced under its 1,200 weight-a-minute limit; a 429 waits and retries."""

    def __init__(self, client: httpx.AsyncClient, weight_per_min: float = 1000.0, sleep=asyncio.sleep):
        self.client, self.weight_per_min, self.sleep = client, weight_per_min, sleep

    async def info(self, body: dict) -> Any:
        for attempt in range(6):
            r = await self.client.post(HL_INFO, json=body)
            if r.status_code == 429:
                await self.sleep(15 * (attempt + 1))
                continue
            r.raise_for_status()
            out = r.json()
            weight = 20 + (len(out) // 20 if isinstance(out, list) and body["type"].startswith("userFills") else 0)
            await self.sleep(weight * 60 / self.weight_per_min)
            return out
        r.raise_for_status()
        raise httpx.HTTPError("Hyperliquid kept answering 429")

    async def leaderboard(self) -> list[dict]:
        r = await self.client.get(HL_LEADERBOARD, timeout=300)
        r.raise_for_status()
        return r.json()["leaderboardRows"]

    async def positions(self, address: str) -> tuple[float, dict[str, dict]]:
        st = await self.info({"type": "clearinghouseState", "user": address})
        out = {}
        for ap in st.get("assetPositions", []):
            p = ap["position"]
            out[p["coin"]] = {"szi": float(p["szi"]), "value": float(p["positionValue"])}
        return float(st.get("marginSummary", {}).get("accountValue", 0) or 0), out


class FuturesMarket:
    """Binance USDⓈ-M futures PUBLIC market data (prices, candles, funding). Never signed, never trades."""

    def __init__(self, client: httpx.AsyncClient, bases: tuple[str, ...] = FUTURES_BASES):
        self.client, self.bases = client, list(bases)

    async def get(self, path: str, params: dict | None = None) -> Any:
        last = None
        for base in list(self.bases):
            try:
                r = await self.client.get(base + path, params=params, timeout=30)
                if r.status_code in (403, 451):  # region refused: try the next base and keep it first from now on
                    last = httpx.HTTPStatusError(f"{base} refused ({r.status_code})", request=r.request, response=r)
                    continue
                r.raise_for_status()
                if base != self.bases[0]:
                    self.bases.remove(base)
                    self.bases.insert(0, base)
                return r.json()
            except httpx.TransportError as e:
                last = e
        raise last or httpx.HTTPError("no futures base reachable")

    async def liquid(self, min_quote_volume: float = MIN_COIN_QUOTE_VOLUME) -> dict[str, str]:
        """Binance base asset -> symbol, for USDT perpetuals trading at least this much a day."""
        rows = await self.get("/fapi/v1/ticker/24hr")
        out = {}
        for t in rows:
            s = t["symbol"]
            if s.endswith("USDT") and float(t.get("quoteVolume", 0)) >= min_quote_volume:
                out[s[:-4]] = s
        return out

    async def price(self, symbol: str) -> float:
        return float((await self.get("/fapi/v1/ticker/price", {"symbol": symbol}))["price"])

    async def klines(self, symbol: str, interval: str, start_ms: int | None = None, limit: int = 500) -> list[list]:
        p = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            p["startTime"] = start_ms
        return await self.get("/fapi/v1/klines", p)

    async def atr(self, symbol: str, days: int = ATR_DAYS) -> float:
        k = await self.klines(symbol, "1d", limit=days + 2)
        k = k[:-1]  # the last daily candle is still forming
        trs = []
        for prev, cur in zip(k[:-1], k[1:], strict=False):
            h, lo, pc = float(cur[2]), float(cur[3]), float(prev[4])
            trs.append(max(h - lo, abs(h - pc), abs(lo - pc)))
        trs = trs[-days:]
        return sum(trs) / len(trs) if trs else float("nan")

    async def funding(self, symbol: str, start: datetime, end: datetime) -> list[float]:
        rows = await self.get("/fapi/v1/fundingRate", {"symbol": symbol, "startTime": int(start.timestamp() * 1000),
                                                      "endTime": int(end.timestamp() * 1000), "limit": 1000})
        return [float(x["fundingRate"]) for x in rows]


# ------------------------------------------------------------------ 1. selection (desktop)
def _window(row: dict, name: str) -> dict:
    for w, perf in row.get("windowPerformances", []):
        if w == name:
            return {k: float(v) for k, v in perf.items()}
    return {"pnl": 0.0, "roi": 0.0, "vlm": 0.0}


async def fill_history(hl: Hyperliquid, address: str) -> list[dict]:
    fills, start = [], 0
    for _ in range(8):  # 8 x 2,000 > the 10,000 fills Hyperliquid keeps
        page = await hl.info({"type": "userFillsByTime", "user": address, "startTime": start, "aggregateByTime": True})
        if not page:
            break
        fills.extend(page)
        if len(page) < 500:
            break
        start = int(page[-1]["time"]) + 1
    return fills


def assess(c: dict, fills: list[dict], first_seen_ms: int | None, liquid: dict[str, str], now: datetime) -> list[str]:
    """Why a candidate does NOT qualify (empty = qualifies). Fills c with the numbers used."""
    why = []
    age = (now.timestamp() * 1000 - first_seen_ms) / 86_400_000 if first_seen_ms else 0
    c["age_days"] = round(age, 1)
    if age < MIN_AGE_DAYS:
        why.append(f"account {age:.0f} days old")
    if not fills:
        return why + ["no fills kept"]
    span = max((fills[-1]["time"] - fills[0]["time"]) / 86_400_000, 1e-9)
    closing = [f for f in fills if float(f.get("closedPnl", 0)) != 0]
    realised = sum(float(f["closedPnl"]) for f in closing)
    vol: dict[str, float] = {}
    for f in fills:
        vol[f["coin"]] = vol.get(f["coin"], 0.0) + float(f["sz"]) * float(f["px"])
    total = sum(vol.values()) or 1.0
    covered = sum(v for coin, v in vol.items() if (binance_base(coin) or "") in liquid)
    c.update({"fills": len(fills), "span_days": round(span, 1), "fills_per_day": round(len(fills) / span, 1),
              "closing_fills": len(closing), "realised_pnl": round(realised, 2),
              "realised_share": round(realised / c["pnl_all"], 3) if c["pnl_all"] else 0.0, "coverage": round(covered / total, 3)})
    if c["fills_per_day"] > MAX_FILLS_PER_DAY:
        why.append(f"{c['fills_per_day']} fills a day")
    if len(closing) < MIN_CLOSING_FILLS:
        why.append(f"{len(closing)} closed trades")
    if span < MIN_SPAN_DAYS:
        why.append(f"{span:.0f} days of history")
    if c["realised_share"] < MIN_REALISED_SHARE:
        why.append(f"realised trading P&L only {c['realised_share']:.0%} of all-time")
    if c["coverage"] < MIN_COVERAGE:
        why.append(f"only {c['coverage']:.0%} of volume in coins Binance futures lists")
    return why


def _category(reason: str) -> str:
    for needle, label in (("days old", "account too new"), ("fills a day", "too many fills a day"),
                          ("closed trades", "too few closed trades"), ("days of history", "history too short"),
                          ("realised", "profit mostly not from trading"), ("volume in coins", "coins Binance futures does not list"),
                          ("no fills", "no fills kept"), ("error", "could not be read")):
        if needle in reason:
            return label
    return reason


async def select_traders(hl: Hyperliquid, market: FuturesMarket, now: datetime | None = None,
                         progress: Callable[[str], None] = print) -> dict:
    now = now or datetime.now(UTC)
    liquid = await market.liquid()
    rows = await hl.leaderboard()
    cands = []
    for r in rows:
        acct = float(r.get("accountValue", 0) or 0)
        at, mo = _window(r, "allTime"), _window(r, "month")
        if acct >= MIN_ACCOUNT and at["pnl"] > 0 and mo["pnl"] > 0:
            cands.append({"address": r["ethAddress"], "name": r.get("displayName"), "account": round(acct),
                          "pnl_all": round(at["pnl"]), "pnl_month": round(mo["pnl"])})
    cands.sort(key=lambda c: -c["pnl_all"])
    progress(f"Leaderboard {len(rows)} rows; {len(cands)} profitable all-time and this month with >= ${MIN_ACCOUNT:,.0f}")
    chosen, rejected, scanned = [], {}, 0
    week_ms = 7 * 86_400_000
    for c in cands[:MAX_SCAN]:
        scanned += 1
        try:
            recent = await hl.info({"type": "userFillsByTime", "user": c["address"],
                                    "startTime": int(now.timestamp() * 1000) - week_ms, "aggregateByTime": True})
            if len(recent) >= 2000 or len(recent) / 7 > MAX_FILLS_PER_DAY:
                rejected["bot or market maker"] = rejected.get("bot or market maker", 0) + 1
                continue
            pf = await hl.info({"type": "portfolio", "user": c["address"]})
            first = next((d.get("accountValueHistory") or [[None]] for n, d in pf if n == "allTime"), [[None]])[0][0]
            why = assess(c, await fill_history(hl, c["address"]), first, liquid, now)
        except Exception as e:  # one unreadable account must not stop the scan
            why = [f"error {type(e).__name__}"]
        if why:
            key = _category(why[0])
            rejected[key] = rejected.get(key, 0) + 1
            continue
        chosen.append(c)
        progress(f"  scanned {scanned}: {len(chosen)} chosen")
    chosen.sort(key=lambda c: -c["realised_pnl"])
    chosen = chosen[:MAX_TRADERS]
    return {"selected_at": now.isoformat(timespec="seconds"), "scanned": scanned, "rejected": rejected,
            "traders": chosen, "coins": sorted(liquid), "rules": rules_dict()}


# ------------------------------------------------------------------ 2. freeze (server, once)
async def load_selection(db) -> dict | None:
    async with db.sessions() as s:
        row = await s.get(StateRow, SELECTION_KEY)
        return dict(row.value) if row else None


async def freeze(db, selection: dict) -> dict:
    if not selection.get("traders"):
        raise ValueError("the selection has no traders")
    if selection.get("rules") != rules_dict():
        raise ValueError("this selection was made under different rules than the code now has")
    async with db.sessions() as s, s.begin():
        if await s.get(StateRow, SELECTION_KEY) is not None:
            raise RuntimeError("Track A's traders are already frozen; the selection cannot be changed")
        value = {**selection, "frozen_at": datetime.now(UTC).isoformat(timespec="seconds")}
        s.add(StateRow(key=SELECTION_KEY, value=value))
    await log_event(db, "INFO", "follow", f"Track A: {len(selection['traders'])} traders frozen; shadow record starts")
    return value


# ------------------------------------------------------------------ 3. record (server, hourly)
def consensus(book: dict[str, tuple[float, dict[str, dict]]], coins: set[str]) -> dict[str, tuple[int, int]]:
    """coin -> (traders meaningfully long, traders meaningfully short), only coins Binance lists."""
    votes: dict[str, list[int]] = {}
    for acct, positions in book.values():
        for coin, p in positions.items():
            base = binance_base(coin)
            if base not in coins or acct <= 0 or p["value"] < MIN_POSITION_SHARE * acct or p["szi"] == 0:
                continue
            v = votes.setdefault(base, [0, 0])
            v[0 if p["szi"] > 0 else 1] += 1
    return {c: (v[0], v[1]) for c, v in votes.items()}


def wants(longs: int, shorts: int) -> str | None:
    if longs >= CONSENSUS_IN and shorts == 0:
        return "BUY"
    if shorts >= CONSENSUS_IN and longs == 0:
        return "SELL"
    return None


def still_on(side: str, longs: int, shorts: int) -> bool:
    return (longs >= CONSENSUS_HOLD and shorts == 0) if side == "BUY" else (shorts >= CONSENSUS_HOLD and longs == 0)


def _slip(price: float, side: str, opening: bool) -> float:
    """Worse price by the slippage: buying pays up, selling gets less."""
    up = (side == "BUY") == opening
    return price * (1 + SLIPPAGE_BPS / 10000) if up else price * (1 - SLIPPAGE_BPS / 10000)


class FollowWatcher:
    def __init__(self, engine, hl: Hyperliquid | None = None, market: FuturesMarket | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.e, self.hl, self.market, self.clock = engine, hl, market, clock

    async def run_forever(self) -> None:
        async with httpx.AsyncClient(timeout=60, headers={"User-Agent": "tradeengine-follow"}) as client:
            self.hl = self.hl or Hyperliquid(client)
            self.market = self.market or FuturesMarket(client)
            while True:
                now = self.clock()
                nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=2)
                await asyncio.sleep((nxt - now).total_seconds())
                try:
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # never let Track A touch trading
                    await self.e.alerts.failed("Track A hourly record", e, component="follow")
                else:
                    await self.e.alerts.succeeded("Track A hourly record", component="follow")

    async def run_once(self) -> dict | None:
        settings = await self.e.settings_store.load()
        if not settings.round4_follow_enabled:
            return None
        sel = await load_selection(self.e.db)
        if sel is None:
            return None
        now = self.clock()
        db = self.e.db
        book = {}
        async with db.sessions() as s, s.begin():
            for t in sel["traders"]:
                acct, pos = await self.hl.positions(t["address"])
                book[t["address"]] = (acct, pos)
                s.add(FollowSnapshotRow(ts=now, address=t["address"], account_value=Decimal(str(acct)), positions=pos))
        coins = set(sel["coins"])
        votes = consensus(book, coins)
        closed = await self._stops(now) + await self._consensus_exits(votes, now)
        await self._clear_waits(votes)
        opened = await self._entries(votes, now, sel)
        equity = await self._mark(now)
        return {"votes": votes, "opened": opened, "closed": closed, "equity": equity}

    async def _open_trades(self) -> list[ShadowTradeRow]:
        async with self.e.db.sessions() as s:
            return list((await s.execute(select(ShadowTradeRow).where(ShadowTradeRow.closed_at.is_(None)))).scalars().all())

    async def _close(self, t: ShadowTradeRow, price: float, when: datetime, reason: str) -> None:
        side, entry, size = t.side, float(t.entry), float(t.size)
        exit_px = _slip(price, side, opening=False)
        notional_in, notional_out = entry * size, exit_px * size
        fees = FEE_RATE * (notional_in + notional_out)
        rates = await self.market.funding(t.symbol, _utc(t.opened_at), when)
        funding = sum(rates) * notional_in * (1 if side == "BUY" else -1)  # positive rate: longs pay shorts
        gross = (exit_px - entry) * size * (1 if side == "BUY" else -1)
        pnl = gross - fees - funding
        async with self.e.db.sessions() as s, s.begin():
            row = await s.get(ShadowTradeRow, t.id)
            row.closed_at, row.exit, row.exit_reason = when, Decimal(str(exit_px)), reason[:128]
            row.fees, row.funding, row.pnl = Decimal(str(round(fees, 6))), Decimal(str(round(funding, 6))), Decimal(str(round(pnl, 6)))
            row.r = round(pnl / float(t.risk_usd), 6)
        log.info("Track A shadow close %s %s: %s, P&L %+.2f", side, t.coin, reason, pnl)

    async def _stops(self, now: datetime) -> list[str]:
        """Touch-the-stop on completed hourly candles since the first full hour after entry."""
        out = []
        for t in await self._open_trades():
            opened_at = _utc(t.opened_at)
            first_hour = opened_at.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            if first_hour + timedelta(hours=1) > now:
                continue
            # checked every hour, so the last 30 days (720 candles) always covers anything not yet looked at
            start = max(first_hour, now.replace(minute=0, second=0, microsecond=0) - timedelta(days=30))
            ks = await self.market.klines(t.symbol, "1h", int(start.timestamp() * 1000), limit=1000)
            stop = float(t.stop)
            for k in ks:
                opened, o, h, lo = datetime.fromtimestamp(k[0] / 1000, UTC), float(k[1]), float(k[2]), float(k[3])
                if opened + timedelta(hours=1) > now:
                    break  # still forming
                if (t.side == "BUY" and lo <= stop) or (t.side == "SELL" and h >= stop):
                    fill = min(o, stop) if t.side == "BUY" else max(o, stop)  # a gap through the stop fills at the open
                    await self._close(t, fill, opened + timedelta(hours=1), "stop")
                    await self._set_waits({**await self._waits(), t.coin: t.side})
                    out.append(f"{t.coin} stop")
                    break
        return out

    async def _waits(self) -> dict[str, str]:
        async with self.e.db.sessions() as s:
            row = await s.get(StateRow, STOPPED_KEY)
            return dict(row.value) if row else {}

    async def _set_waits(self, waits: dict[str, str]) -> None:
        async with self.e.db.sessions() as s, s.begin():
            row = await s.get(StateRow, STOPPED_KEY)
            if row is None:
                s.add(StateRow(key=STOPPED_KEY, value=waits))
            else:
                row.value = waits

    async def _clear_waits(self, votes: dict) -> None:
        waits = await self._waits()
        keep = {c: side for c, side in waits.items() if wants(*votes.get(c, (0, 0))) == side}
        if keep != waits:
            await self._set_waits(keep)

    async def _consensus_exits(self, votes: dict, now: datetime) -> list[str]:
        out = []
        for t in await self._open_trades():
            longs, shorts = votes.get(t.coin, (0, 0))
            if not still_on(t.side, longs, shorts):
                await self._close(t, await self.market.price(t.symbol), now, f"consensus broke ({longs} long, {shorts} short)")
                out.append(f"{t.coin} consensus")
        return out

    async def _equity(self, now: datetime, prices: dict[str, float] | None = None) -> float:
        async with self.e.db.sessions() as s:
            realised = (await s.execute(select(func.coalesce(func.sum(ShadowTradeRow.pnl), 0)))).scalar_one()
        eq = START_EQUITY + float(realised or 0)
        for t in await self._open_trades():
            px = (prices or {}).get(t.symbol) or await self.market.price(t.symbol)
            eq += (px - float(t.entry)) * float(t.size) * (1 if t.side == "BUY" else -1)
        return eq

    async def _entries(self, votes: dict, now: datetime, sel: dict) -> list[str]:
        db = self.e.db
        async with db.sessions() as s:
            if await s.get(StateRow, KILLED_KEY) is not None:
                return []
            rows = (await s.execute(select(ShadowEquityRow).order_by(ShadowEquityRow.ts))).scalars().all()
        open_now = await self._open_trades()
        held = {t.coin for t in open_now} | set(await self._waits())
        equity = await self._equity(now)
        peak = max([START_EQUITY] + [float(r.equity) for r in rows] + [equity])
        if equity <= peak * (1 - MAX_DD_PCT / 100):
            async with db.sessions() as s, s.begin():
                s.add(StateRow(key=KILLED_KEY, value={"at": now.isoformat(), "equity": equity, "peak": peak}))
            await log_event(db, "WARNING", "follow", f"Track A shadow drawdown {100 * (1 - equity / peak):.1f}%: test failed, no new entries")
            return []
        day = [float(r.equity) for r in rows if _utc(r.ts).date() == now.date()]
        if day and equity <= day[0] * (1 - DAILY_LOSS_PCT / 100):
            return []
        out = []
        for coin in sorted(votes):
            side = wants(*votes[coin])
            if side is None or coin in held or len(open_now) + len(out) >= MAX_OPEN:
                continue
            symbol = coin + "USDT"
            price = await self.market.price(symbol)
            atr = await self.market.atr(symbol)
            if not (math.isfinite(atr) and atr > 0):
                continue
            entry = _slip(price, side, opening=True)
            stop = entry - STOP_ATR * atr if side == "BUY" else entry + STOP_ATR * atr
            risk = equity * RISK_PCT
            size = risk / (STOP_ATR * atr)
            notional_open = sum(float(t.entry) * float(t.size) for t in open_now)
            if notional_open + entry * size > MAX_LEVERAGE * equity:
                continue
            longs, shorts = votes[coin]
            async with db.sessions() as s, s.begin():
                s.add(ShadowTradeRow(coin=coin, symbol=symbol, side=side, opened_at=now, entry=Decimal(str(entry)),
                                     stop=Decimal(str(stop)), size=Decimal(str(size)), risk_usd=Decimal(str(round(risk, 6))),
                                     reason=f"{longs} long / {shorts} short of {len(sel['traders'])} followed"))
            out.append(f"{side} {coin}")
        return out

    async def _mark(self, now: datetime) -> float:
        eq = await self._equity(now)
        async with self.e.db.sessions() as s, s.begin():
            s.add(ShadowEquityRow(ts=now, equity=Decimal(str(round(eq, 6)))))
        return eq


def _utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


# ------------------------------------------------------------------ status and the one-look score
@dataclass
class FollowStats:
    since: str | None
    days: float
    open: int
    closed: int
    pnl: float
    profit_factor: float | None
    max_dd_pct: float
    bootstrap_lower_bound: float | None
    killed: bool


async def stats(db, now: datetime | None = None) -> FollowStats | None:
    sel = await load_selection(db)
    if sel is None:
        return None
    now = now or datetime.now(UTC)
    async with db.sessions() as s:
        trades = (await s.execute(select(ShadowTradeRow))).scalars().all()
        eq = [float(r.equity) for r in (await s.execute(select(ShadowEquityRow).order_by(ShadowEquityRow.ts))).scalars().all()]
        killed = await s.get(StateRow, KILLED_KEY) is not None
    closed = [t for t in trades if t.closed_at is not None]
    rs = [float(t.r) for t in closed]
    wins, losses = sum(r for r in rs if r > 0), -sum(r for r in rs if r <= 0)
    pf = (wins / losses) if losses > 0 else (math.inf if wins > 0 else None)
    peak, dd = START_EQUITY, 0.0
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, 100 * (1 - v / peak))
    since = datetime.fromisoformat(sel["frozen_at"])
    return FollowStats(sel["frozen_at"][:10], (now - since).total_seconds() / 86400, len(trades) - len(closed), len(closed),
                       round(sum(float(t.pnl) for t in closed), 2), pf, round(dd, 2),
                       round(bootstrap_lower_bound(rs, RULES), 4) if len(rs) >= 2 else None, killed)


def summary_line(st: FollowStats) -> str:
    pf = "n/a" if st.profit_factor is None else ("all wins" if math.isinf(st.profit_factor) else f"{st.profit_factor:.2f}")
    return (f"Track A shadow (since {st.since}): {st.open} open, {st.closed} closed, P&L {st.pnl:+.2f}, "
            f"profit factor {pf}, drawdown {st.max_dd_pct:.1f}%" + (" - STOPPED at 10% drawdown" if st.killed else ""))


def judge(st: FollowStats) -> tuple[bool, list[str]]:
    why = []
    if st.killed or st.max_dd_pct >= RULES.holdout_max_dd_pct:
        why.append(f"drawdown {st.max_dd_pct:.1f}%, limit {RULES.holdout_max_dd_pct}%")
    if st.closed < RULES.holdout_min_trades:
        why.append(f"{st.closed} closed trades, needs {RULES.holdout_min_trades}")
    if (st.profit_factor or 0) < RULES.holdout_min_pf:
        why.append(f"profit factor {st.profit_factor or 0:.2f}, needs {RULES.holdout_min_pf}")
    if st.bootstrap_lower_bound is None or st.bootstrap_lower_bound <= 0:
        why.append(f"statistical check: 5th-percentile average R {st.bootstrap_lower_bound}, needs > 0")
    return not why, why


async def score(db, now: datetime | None = None) -> tuple[bool, list[str], FollowStats]:
    now = now or datetime.now(UTC)
    async with db.sessions() as s:
        done = await s.get(StateRow, SCORE_KEY)
    if done is not None:
        raise RuntimeError(f"Track A was already scored on {done.value.get('at')} ({done.value.get('verdict')}); the result is locked")
    st = await stats(db, now)
    if st is None:
        raise RuntimeError("Track A has no frozen selection yet")
    if st.days < MIN_RECORD_DAYS or st.closed < RULES.holdout_min_trades:
        if not st.killed:
            raise RuntimeError(f"Too early: {st.days:.0f} days and {st.closed} closed shadow trades so far; the score needs "
                               f"{MIN_RECORD_DAYS} days and {RULES.holdout_min_trades} trades")
    ok, why = judge(st)
    async with db.sessions() as s, s.begin():
        s.add(StateRow(key=SCORE_KEY, value={"at": now.isoformat(timespec="seconds"), "verdict": "PASS" if ok else "FAIL",
                                             "reasons": why, "stats": {k: (None if isinstance(v, float) and math.isinf(v) else v)
                                                                        for k, v in asdict(st).items()}}))
    return ok, why, st
