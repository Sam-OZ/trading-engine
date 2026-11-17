"""Round 4 investigation probe: READ-ONLY. Places no orders, changes nothing, prints no keys.

Answers four questions before any round 4 code is written (v2: paced, skips bots cheaply, scans wider):
  1. Which Hyperliquid traders look "good" on the public leaderboard, and are they directional or hedged?
  2. How much trade history does Hyperliquid let us download for each (the API keeps only the last 10,000 fills)?
  3. Which of their coins can our current Binance demo account trade (spot demo = long only; futures demo = long/short)?
  4. Does our current Binance demo API key also work on Binance's FUTURES demo (demo-fapi.binance.com)?

Run from your Mac (Hyperliquid and Binance both work from Australia; the keys are loaded only for this process):
  ./scripts/with-secrets.sh uv run python scripts/probe_round4.py
Writes data/reports/round4-probe-<time>.json and prints a short summary.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx

HL_INFO = "https://api.hyperliquid.xyz/info"
HL_LEADERBOARD = "https://stats-data.hyperliquid.xyz/Mainnet/leaderboard"
SPOT_DEMO = "https://demo-api.binance.com"
FUT_DEMO = "https://demo-fapi.binance.com"

MIN_ACCOUNT = 100_000  # USD: ignore tiny accounts
MAX_SCAN = 400  # candidates looked at, best all-time P&L first
WANT = 40  # stop once this many look copyable
MAX_FILLS_PER_DAY = 50  # more = market maker / bot: cannot be followed an hour late
MIN_CLOSING_FILLS = 100
MIN_SPAN_DAYS = 90
# Hyperliquid allows 1,200 request weight a minute per IP; fills cost 20 + 1 per 20 returned. Pace on that.
WEIGHT_PER_MIN = 1000


def ms_to_date(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d") if ms else None


def window(row: dict, name: str) -> dict:
    for w, perf in row.get("windowPerformances", []):
        if w == name:
            return {k: float(v) for k, v in perf.items()}
    return {"pnl": 0.0, "roi": 0.0, "vlm": 0.0}


def hl(client: httpx.Client, body: dict):
    """One Hyperliquid info call, paced by its weight; a 429 waits and retries."""
    for attempt in range(6):
        r = client.post(HL_INFO, json=body)
        if r.status_code == 429:
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        out = r.json()
        weight = 20 + (len(out) // 20 if isinstance(out, list) and body["type"].startswith("userFills") else 0)
        time.sleep(weight * 60 / WEIGHT_PER_MIN)
        return out
    r.raise_for_status()


def recent_fill_rate(client: httpx.Client, addr: str) -> int:
    """Fills in the last 7 days (2,000 = the page is full: a bot or market maker). One cheap call."""
    since = int(time.time() * 1000) - 7 * 86_400_000
    return len(hl(client, {"type": "userFillsByTime", "user": addr, "startTime": since, "aggregateByTime": True}) or [])


def fill_history(client: httpx.Client, addr: str) -> dict:
    """Page forward from the oldest fill Hyperliquid still keeps."""
    fills, start = [], 0
    for _ in range(8):  # 8 x 2,000 > the 10,000 Hyperliquid keeps
        page = hl(client, {"type": "userFillsByTime", "user": addr, "startTime": start, "aggregateByTime": True})
        if not page:
            break
        fills.extend(page)
        if len(page) < 500:
            break
        start = int(page[-1]["time"]) + 1
    if not fills:
        return {"fills": 0}
    first, last = fills[0]["time"], fills[-1]["time"]
    span_days = max((last - first) / 86_400_000, 1e-9)
    closed = [f for f in fills if float(f.get("closedPnl", 0)) != 0]
    coins: dict[str, float] = {}
    for f in fills:
        coins[f["coin"]] = coins.get(f["coin"], 0.0) + float(f["sz"]) * float(f["px"])
    total = sum(coins.values()) or 1.0
    return {
        "fills": len(fills),
        "first": ms_to_date(first),
        "last": ms_to_date(last),
        "span_days": round(span_days, 1),
        "fills_per_day": round(len(fills) / span_days, 1),
        "closing_fills": len(closed),
        "closed_pnl": round(sum(float(f["closedPnl"]) for f in closed), 2),
        "top_coins": {c: round(v / total, 3) for c, v in sorted(coins.items(), key=lambda kv: -kv[1])[:8]},
    }


def positions(client: httpx.Client, addr: str) -> dict:
    st = hl(client, {"type": "clearinghouseState", "user": addr})
    longs = shorts = 0.0
    n = 0
    for ap in st.get("assetPositions", []):
        p = ap["position"]
        szi, val = float(p["szi"]), float(p["positionValue"])
        n += 1
        if szi > 0:
            longs += val
        else:
            shorts += val
    acct = float(st.get("marginSummary", {}).get("accountValue", 0) or 0)
    gross = longs + shorts
    return {
        "open_positions": n,
        "long_usd": round(longs),
        "short_usd": round(shorts),
        "net_long_share": round((longs - shorts) / gross, 2) if gross else None,  # +1 all long, -1 all short
        "leverage": round(gross / acct, 2) if acct else None,
    }


def account_age(client: httpx.Client, addr: str) -> dict:
    pf = hl(client, {"type": "portfolio", "user": addr})
    for name, data in pf:
        if name == "allTime":
            hist = data.get("accountValueHistory") or []
            pnl = data.get("pnlHistory") or []
            return {"first_seen": ms_to_date(hist[0][0]) if hist else None,
                    "pnl_points": len(pnl)}
    return {}


def binance_symbols(client: httpx.Client) -> dict:
    out = {}
    for name, url in (("spot_demo", f"{SPOT_DEMO}/api/v3/exchangeInfo"), ("futures_demo", f"{FUT_DEMO}/fapi/v1/exchangeInfo")):
        try:
            r = client.get(url, timeout=30)
            r.raise_for_status()
            syms = r.json()["symbols"]
            out[name] = sorted(s["baseAsset"] for s in syms if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING")
        except Exception as e:  # e.g. the spot demo maintenance on 17 Nov
            out[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    return out


def futures_demo_key_check(client: httpx.Client) -> dict:
    key, secret = os.environ.get("BINANCE_API_KEY"), os.environ.get("BINANCE_API_SECRET")
    if not key or not secret:
        return {"checked": False, "why": "keys not loaded (run through scripts/with-secrets.sh)"}
    q = urlencode({"timestamp": int(time.time() * 1000), "recvWindow": 5000})
    sig = hmac.new(secret.encode(), q.encode(), hashlib.sha256).hexdigest()
    try:
        r = client.get(f"{FUT_DEMO}/fapi/v2/balance?{q}&signature={sig}", headers={"X-MBX-APIKEY": key}, timeout=20)
    except Exception as e:
        return {"checked": True, "works": False, "why": f"{type(e).__name__}"}
    if r.status_code != 200:
        try:
            body = r.json()
            why = f"HTTP {r.status_code}, code {body.get('code')}: {body.get('msg')}"
        except Exception:
            why = f"HTTP {r.status_code}"
        return {"checked": True, "works": False, "why": why}
    usdt = next((b for b in r.json() if b.get("asset") == "USDT"), {})
    return {"checked": True, "works": True, "usdt_balance": usdt.get("balance"), "usdt_available": usdt.get("availableBalance")}


def hl_to_binance(coin: str) -> str:
    return "1000" + coin[1:] if coin.startswith("k") and coin[1:].isupper() else coin


def main() -> None:
    started = datetime.now(UTC)
    report: dict = {"run_at": started.isoformat(timespec="seconds")}
    with httpx.Client(timeout=60, headers={"User-Agent": "tradeengine-round4-probe"}) as client:
        print("Binance demo: symbols and futures-demo key check ...", file=sys.stderr)
        report["binance"] = binance_symbols(client)
        report["futures_demo_key"] = futures_demo_key_check(client)

        print("Hyperliquid leaderboard (about 40 MB) ...", file=sys.stderr)
        rows = client.get(HL_LEADERBOARD, timeout=180).json()["leaderboardRows"]
        report["leaderboard_rows"] = len(rows)
        cands = []
        for r in rows:
            acct = float(r.get("accountValue", 0) or 0)
            at, mo, wk = window(r, "allTime"), window(r, "month"), window(r, "week")
            if acct >= MIN_ACCOUNT and at["pnl"] > 0 and mo["pnl"] > 0:
                cands.append({"address": r["ethAddress"], "name": r.get("displayName"), "account": round(acct),
                              "pnl_all": round(at["pnl"]), "roi_all": round(at["roi"], 3),
                              "pnl_month": round(mo["pnl"]), "pnl_week": round(wk["pnl"]), "vlm_all": round(at["vlm"])})
        report["candidates_profitable_all_time_and_month"] = len(cands)
        cands.sort(key=lambda c: -c["pnl_all"])

        spot = set(report["binance"]["spot_demo"]) if isinstance(report["binance"]["spot_demo"], list) else set()
        fut = set(report["binance"]["futures_demo"]) if isinstance(report["binance"]["futures_demo"], list) else set()
        scanned, bots, copyable = [], 0, []
        for i, c in enumerate(cands[:MAX_SCAN], 1):
            if len(copyable) >= WANT:
                break
            try:
                week = recent_fill_rate(client, c["address"])
                c["fills_last_7d"] = week
                if week >= 2000 or week / 7 > MAX_FILLS_PER_DAY:
                    bots += 1
                    c["verdict"] = "bot or market maker (too many fills)"
                    scanned.append(c)
                    continue
                c.update(account_age(client, c["address"]))
                c["now"] = positions(client, c["address"])
                c["history"] = fill_history(client, c["address"])
                h = c["history"]
                tc = h.get("top_coins", {})
                c["share_on_binance_futures_demo"] = round(sum(v for k, v in tc.items() if hl_to_binance(k) in fut), 2) if fut else None
                c["share_on_binance_spot_demo"] = round(sum(v for k, v in tc.items() if k in spot), 2) if spot else None
                c["realised_share_of_all_time"] = round(h.get("closed_pnl", 0) / c["pnl_all"], 2) if c["pnl_all"] else None
                why = []
                if h.get("fills_per_day", 0) > MAX_FILLS_PER_DAY:
                    why.append("too many fills a day")
                if h.get("closing_fills", 0) < MIN_CLOSING_FILLS:
                    why.append("too few closed trades")
                if h.get("span_days", 0) < MIN_SPAN_DAYS:
                    why.append("history too short")
                c["verdict"] = "copyable" if not why else "; ".join(why)
                if not why:
                    copyable.append(c)
            except Exception as e:
                c["error"] = f"{type(e).__name__}: {str(e)[:120]}"
                c["verdict"] = "error"
            scanned.append(c)
            print(f"  scanned {i}, copyable so far {len(copyable)}, bots {bots}", file=sys.stderr)
        report["scanned"] = len(scanned)
        report["bots_or_market_makers"] = bots
        report["copyable"] = len(copyable)
        report["shortlist"] = scanned

    out = Path("data/reports") / f"round4-probe-{started:%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(f"\nSaved {out}")
    b = report["binance"]
    print(f"Binance spot demo USDT pairs: {len(b['spot_demo']) if isinstance(b['spot_demo'], list) else b['spot_demo']}")
    print(f"Binance futures demo USDT pairs: {len(b['futures_demo']) if isinstance(b['futures_demo'], list) else b['futures_demo']}")
    print(f"Futures demo with our current key: {report['futures_demo_key']}")
    print(f"Leaderboard rows: {report['leaderboard_rows']}; profitable all-time AND this month, account >= ${MIN_ACCOUNT:,}: "
          f"{report['candidates_profitable_all_time_and_month']}")
    print(f"Scanned {report['scanned']}: bots/market makers {report['bots_or_market_makers']}, copyable {report['copyable']}")
    print("Copyable (all-time PnL | realised share | history | fills/day | closed trades | on Binance futures | net long now):")
    for c in report["shortlist"]:
        if c.get("verdict") != "copyable":
            continue
        h, n = c.get("history", {}), c.get("now", {})
        print(f"  {c['address'][:10]}… {(c.get('name') or '')[:16]:<16} ${c['pnl_all']:>13,} | {c.get('realised_share_of_all_time')} | "
              f"{h.get('first')}→{h.get('last')} | {h.get('fills_per_day')} | {h.get('closing_fills')} | "
              f"{c.get('share_on_binance_futures_demo')} | {n.get('net_long_share')}")


if __name__ == "__main__":
    main()
