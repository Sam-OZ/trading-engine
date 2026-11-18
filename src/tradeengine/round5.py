"""Round 5: C2's short side on Binance USDⓈ-M perpetual futures. Approved by Sam 18 Nov 2026 (plan: Claude Docs
"Round 5 plan - testing C2's short side on Binance futures"). The constants below ARE the pre-registration.

- The rule is C2 turned upside down and NOTHING is re-chosen: at each daily close (00:00 UTC) sell short when the
  close is below the close L days ago (BTC L=14, ETH L=56: the forward test's own settings); buy back at the first
  close where that return is >= 0; stop 3 x ATR(20) above, trailed 4 x ATR above the lowest close; 0.5% risk.
- Costs: 0.05% a side (Binance regular taker), 1-1.5 bp spread, 2 bps slippage, and the REAL funding rate at
  every funding time while a short is open (shorts receive a positive rate and pay a negative one).
- Two looks, in order: development 1 Mar 2020 - 31 Oct 2023 (may be re-run, every run counted); unseen
  1 Nov 2023 - 31 Oct 2026, once, only after a development PASS with the same fingerprint, then locked.
- Pass (each look): >= 40 trades, profit factor >= 1.2, drawdown < 10%, 5th percentile of 10,000 seeded bootstrap
  means of trade R > 0. Information only (not the verdict): C2 long on spot plus this short side together.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from . import portfolio as pf_mod
from . import strategies as strat_mod
from . import trend as trend_mod
from .backtest import cost_for
from .db import StateRow
from .history import frame_to_bars, history_path, load_h1, read_frame, write_bars
from .models import AssetKind, Side, Venue
from .portfolio import Leg, Limits, RateBook, portfolio_metrics, simulate, symbol_stats
from .round2 import START_EQUITY, Plan, Rules, bootstrap_lower_bound, record_holdout
from .trend import TimeSeriesMomentumFastD1, prep_key

FAPI = "https://fapi.binance.com"
SOURCE = "binance_futures"
# canonical perp symbol -> (Binance futures symbol, the spot symbol C2 trades)
PERPS = {"BTC/USDT-PERP": ("BTCUSDT", "BTC/USDT"), "ETH/USDT-PERP": ("ETHUSDT", "ETH/USDT")}
LISTED = {"BTCUSDT": pd.Timestamp("2019-09-08", tz="UTC"), "ETHUSDT": pd.Timestamp("2019-11-27", tz="UTC")}
DEV_FROM = pd.Timestamp("2020-03-01", tz="UTC")
DEV_END = pd.Timestamp("2023-11-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-11-01", tz="UTC")
WARMUP_DAYS = 70  # 56-day lookback + ATR(20) need this much history before the first scored close
LOOKBACK = {"BTC/USDT-PERP": 14, "ETH/USDT-PERP": 56}  # = C2_PARAMS of the forward test
STOP_ATR, TRAIL_ATR = 3.0, 4.0
RULES = Rules(dev_min_pf=1.2, dev_max_dd_pct=10.0, dev_min_trades=40, holdout_min_pf=1.2, holdout_max_dd_pct=10.0,
              holdout_min_trades=40, bootstrap=True, bootstrap_samples=10_000, bootstrap_pct=5.0)
STRATEGY = "c2_short"
DEV_KEY = "round5_dev:" + STRATEGY


def plan() -> Plan:
    return Plan(5, list(PERPS), {"crypto": DEV_FROM, "fx": DEV_FROM}, DEV_END, HOLDOUT_END, RULES, {}, "round3_walk_forward")


def params_for(symbol: str, side: str = "short") -> dict:
    lb = LOOKBACK[symbol] if symbol in LOOKBACK else LOOKBACK[symbol + "-PERP"]
    return {"lookback": lb, "stop_atr": STOP_ATR, "trail_atr": TRAIL_ATR, "side": side}


# ------------------------------------------------------------------ the strategy
@dataclass
class C2Mirror(TimeSeriesMomentumFastD1):
    """C2 with one direction per leg: params["side"] "short" = only the short half (needs a market that can
    short), "long" = only C2's long half (spot). Exits are C2's, unchanged."""

    name: str = STRATEGY
    prep_keys: tuple = ("lookback",)
    grid: dict = field(default_factory=lambda: {"lookback": [14, 56], "stop_atr": [3.0], "trail_atr": [4.0]})

    def signal_at(self, cols, i, p, allow_short):
        side = p.get("side", "short")
        sig = super().signal_at(cols, i, p, allow_short or side == "short")
        if sig is None:
            return None
        if side == "short" and not allow_short:
            return None  # a short leg on a market that cannot short: never
        return sig if (sig.side is Side.SELL) == (side == "short") else None


# ------------------------------------------------------------------ data: hourly candles and funding history
def funding_path(data_dir: Path, symbol: str) -> Path:
    return data_dir / "history" / SOURCE / f"{PERPS[symbol][0]}_funding.csv"


async def _get(client: httpx.AsyncClient, path: str, params: dict, sleep=asyncio.sleep) -> list:
    for wait in (2, 5, 15, 30, 60, None):
        r = await client.get(FAPI + path, params=params, timeout=30)
        if r.status_code in (418, 429, 500, 502, 503, 504):
            if wait is None:
                r.raise_for_status()
            await sleep(float(r.headers.get("Retry-After", wait)))
            continue
        r.raise_for_status()
        await sleep(0.25)  # well inside Binance's public weight limit
        return r.json()
    raise httpx.HTTPError(f"{path}: Binance kept refusing")


async def download(data_dir: Path, until: pd.Timestamp | None = None, progress: Callable[[str], None] = print,
                   client: httpx.AsyncClient | None = None, sleep=asyncio.sleep) -> dict[str, str]:
    """Hourly candles and funding rates for both perpetuals, from listing to `until` (default now). Resumable:
    only what is missing after the last saved candle / payment is fetched."""
    until = until or pd.Timestamp.now(tz="UTC").floor("h")
    own = client is None
    client = client or httpx.AsyncClient(headers={"User-Agent": "tradeengine-round5"})
    out = {}
    try:
        for sym, (fsym, _) in PERPS.items():
            path = history_path(data_dir, SOURCE, sym, "H1")
            have = read_frame(path)
            start = have.index[-1] + pd.Timedelta(hours=1) if have is not None and len(have) else LISTED[fsym]
            rows = []
            while start < until:
                k = await _get(client, "/fapi/v1/klines", {"symbol": fsym, "interval": "1h", "limit": 1500,
                                                          "startTime": int(start.timestamp() * 1000)}, sleep)
                k = [x for x in k if pd.Timestamp(x[0], unit="ms", tz="UTC") < until and x[6] < until.timestamp() * 1000]
                if not k:
                    break
                rows += k
                start = pd.Timestamp(k[-1][0], unit="ms", tz="UTC") + pd.Timedelta(hours=1)
                if len(rows) % 15000 < 1500:
                    progress(f"  {sym}: candles to {start:%Y-%m-%d}")
            if rows:
                df = pd.DataFrame([[pd.Timestamp(x[0], unit="ms", tz="UTC"), *map(float, x[1:6])] for x in rows],
                                  columns=["open_time", "open", "high", "low", "close", "volume"]).set_index("open_time")
                write_bars(path, frame_to_bars(df, sym, "H1"))
            fpath = funding_path(data_dir, sym)
            fhave = read_funding(fpath)
            fstart = (fhave.index[-1] + pd.Timedelta(minutes=1)) if fhave is not None and len(fhave) else LISTED[fsym]
            frows = []
            while fstart < until:
                f = await _get(client, "/fapi/v1/fundingRate", {"symbol": fsym, "limit": 1000,
                                                               "startTime": int(fstart.timestamp() * 1000),
                                                               "endTime": int(until.timestamp() * 1000)}, sleep)
                if not f:
                    break
                frows += f
                fstart = pd.Timestamp(int(f[-1]["fundingTime"]), unit="ms", tz="UTC") + pd.Timedelta(minutes=1)
                if len(f) < 1000:
                    break
            if frows:
                new = pd.Series({pd.Timestamp(int(x["fundingTime"]), unit="ms", tz="UTC").floor("min"): float(x["fundingRate"])
                                 for x in frows}, name="rate")
                merged = new if fhave is None else pd.concat([fhave, new])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                fpath.parent.mkdir(parents=True, exist_ok=True)
                merged.rename_axis("time").to_csv(fpath)
            h = read_frame(path)
            fr = read_funding(fpath)
            out[sym] = (f"{len(h):,} hourly candles {h.index[0]:%Y-%m-%d} to {h.index[-1]:%Y-%m-%d}; "
                        f"{len(fr):,} funding payments {fr.index[0]:%Y-%m-%d} to {fr.index[-1]:%Y-%m-%d}")
            progress(f"{sym}: {out[sym]}")
    finally:
        if own:
            await client.aclose()
    return out


def read_funding(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return pd.Series(df["rate"].to_numpy(dtype=float), index=pd.to_datetime(df["time"], utc=True), name="rate").sort_index()


@dataclass
class PerpData:
    h1: pd.DataFrame
    funding: pd.Series
    note: str


def load_perps(data_dir: Path) -> dict[str, PerpData]:
    """Both perpetuals up to HOLDOUT_END, refusing anything incomplete: a late start, a gap over 4 days in the
    candles, or funding missing for more than a day while the contract traded."""
    out, problems = {}, []
    for sym, (fsym, _) in PERPS.items():
        h1 = read_frame(history_path(data_dir, SOURCE, sym, "H1"))
        fr = read_funding(funding_path(data_dir, sym))
        if h1 is None or fr is None or h1.empty or fr.empty:
            problems.append(f"{sym}: no futures history yet (run `tradeengine round5 --download`)")
            continue
        h1 = h1[h1.index < HOLDOUT_END]
        fr = fr[fr.index <= HOLDOUT_END]
        if h1.index[0] > DEV_FROM - pd.Timedelta(days=WARMUP_DAYS):
            problems.append(f"{sym}: candles start {h1.index[0]:%Y-%m-%d}, too late for {DEV_FROM:%Y-%m-%d}")
        if h1.index[-1] < HOLDOUT_END - pd.Timedelta(hours=2):
            problems.append(f"{sym}: candles end {h1.index[-1]:%Y-%m-%d %H:%M}; download again")
        gap = h1.index.to_series().diff().max()
        if gap > pd.Timedelta(days=4):
            problems.append(f"{sym}: a {gap} gap in the candles")
        fgap = fr.index.to_series().diff().max()
        if fgap > pd.Timedelta(days=1) or fr.index[0] > LISTED[fsym] + pd.Timedelta(days=2) or fr.index[-1] < HOLDOUT_END - pd.Timedelta(hours=9):
            problems.append(f"{sym}: funding history incomplete (largest gap {fgap}, {fr.index[0]:%Y-%m-%d} to {fr.index[-1]:%Y-%m-%d})")
        out[sym] = PerpData(h1, fr, f"{sym}: {len(h1):,} hourly candles {h1.index[0]:%Y-%m-%d} to {h1.index[-1]:%Y-%m-%d}; "
                                    f"{len(fr):,} funding payments, average {fr.mean() * 3 * 365:+.1%} a year")
    if problems:
        raise SystemExit("Round 5 data not ready:\n  " + "\n  ".join(problems))
    return out


# ------------------------------------------------------------------ simulation
def _short_legs(data: dict[str, PerpData], overrides) -> list[Leg]:
    s = C2Mirror()
    legs = []
    for sym, d in data.items():
        p = params_for(sym)
        legs.append(Leg(sym, AssetKind.CRYPTO_CFD, Venue.BINANCE, sym.split("/")[0], "USDT", cost_for(sym, overrides),
                        {prep_key(s, p): s.prepare(d.h1, p, True)}, [(DEV_FROM, HOLDOUT_END, p)],
                        prep_of=lambda q, st=s: prep_key(st, q), funding=d.funding))
    return legs


def _long_legs(data_dir: Path, overrides) -> tuple[list[Leg], list[str]]:
    s = C2Mirror()
    legs, notes = [], []
    for sym, (_, spot) in PERPS.items():
        h1, note = load_h1(data_dir, spot, True)
        notes.append(note)
        if h1 is None:
            continue
        p = params_for(sym, "long")
        legs.append(Leg(spot, AssetKind.CRYPTO_SPOT, Venue.BINANCE, spot.split("/")[0], "USDT", cost_for(spot, overrides),
                        {prep_key(s, p): s.prepare(h1[h1.index < HOLDOUT_END], p, True)}, [(DEV_FROM, HOLDOUT_END, p)],
                        prep_of=lambda q, st=s: prep_key(st, q)))
    return legs, notes


def run(legs: list[Leg], settings, start: pd.Timestamp, end: pd.Timestamp):
    rates = RateBook({lg.symbol: next(iter(lg.frames.values()))["close"] for lg in legs})
    return simulate(C2Mirror(), legs, rates, start, end, Limits.from_settings(settings), START_EQUITY)


def summary(res) -> dict:
    m = portfolio_metrics(res)
    d = asdict(m)
    d["profit_factor"] = None if math.isinf(m.profit_factor) else round(m.profit_factor, 3)
    d["bootstrap_lower_bound"] = round(bootstrap_lower_bound([t.r for t in res.trades], RULES), 4) if len(res.trades) >= 2 else None
    d["per_symbol"] = {s: [n, (None if math.isinf(p) else round(p, 3)), round(r, 2)] for s, (n, p, r) in symbol_stats(res.trades).items()}
    d["funding_paid"] = round(sum(t.funding for t in res.trades), 2)  # negative = received
    d["fees_and_funding"] = round(sum(t.costs for t in res.trades), 2)
    d["shorts"] = sum(1 for t in res.trades if t.side == Side.SELL.value)
    d["halted_at"] = res.halted_at
    d["rejections"] = dict(res.rejections)
    return d


def judge(m: dict, which: str) -> tuple[bool, list[str]]:
    r = RULES
    min_trades = r.dev_min_trades if which == "dev" else r.holdout_min_trades
    min_pf = r.dev_min_pf if which == "dev" else r.holdout_min_pf
    max_dd = r.dev_max_dd_pct if which == "dev" else r.holdout_max_dd_pct
    why = []
    if m["trades"] < min_trades:
        why.append(f"{m['trades']} trades, needs {min_trades}")
    pf = math.inf if m["profit_factor"] is None and m["trades"] else (m["profit_factor"] or 0.0)
    if pf < min_pf:
        why.append(f"profit factor {pf:.2f}, needs {min_pf}")
    if m["max_drawdown_pct"] >= max_dd or m["halted_at"]:
        why.append(f"drawdown {m['max_drawdown_pct']:.1f}%, limit {max_dd}%")
    lb = m["bootstrap_lower_bound"]
    if lb is None or lb <= 0:
        why.append(f"statistical check: 5th-percentile average R {lb}, needs > 0")
    return not why, why


def fingerprint(settings, data: dict[str, PerpData]) -> str:
    from . import round5 as me

    parts = [inspect.getsource(m) for m in (me, pf_mod, trend_mod, strat_mod)]
    for sym, d in sorted(data.items()):
        parts.append(f"{sym}:{int(pd.util.hash_pandas_object(d.h1, index=True).sum())}:"
                     f"{int(pd.util.hash_pandas_object(d.funding, index=True).sum())}")
    parts.append(json.dumps({s: asdict(cost_for(s, settings.cost_model)) for s in [*PERPS, *(v[1] for v in PERPS.values())]}, sort_keys=True))
    parts.append(json.dumps(asdict(Limits.from_settings(settings)), sort_keys=True))
    parts.append(json.dumps({"rules": asdict(RULES), "lookback": LOOKBACK, "dates": [str(DEV_FROM), str(DEV_END), str(HOLDOUT_END)]}))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _row(name: str, m: dict) -> str:
    pf = "inf" if m["profit_factor"] is None and m["trades"] else f"{(m['profit_factor'] or 0):.2f}"
    lb = "n/a" if m["bootstrap_lower_bound"] is None else f"{m['bootstrap_lower_bound']:+.3f}"
    return (f"| {name} | {m['trades']} | {m['win_rate']:.0%} | {pf} | {m['expectancy_r']:+.2f} | {m['max_drawdown_pct']:.1f}% | "
            f"{m['return_pct']:+.1f}% | {lb} | {m.get('funding_paid', 0):+,.2f} |")


def render(which: str, ok: bool, why: list[str], m: dict, fp: str, notes: list[str], combined: dict | None = None) -> str:
    start, end = (DEV_FROM, DEV_END) if which == "dev" else (DEV_END, HOLDOUT_END)
    lines = [
        f"# Round 5 ({'development' if which == 'dev' else 'unseen, one look'}): C2's short side, {datetime.now(UTC):%Y-%m-%d %H:%M} UTC",
        "",
        f"**Verdict: {'PASS' if ok else 'FAIL'}**" + ("" if ok else " - " + "; ".join(why)),
        "",
        f"Daily closes {start:%d %b %Y} to {end - pd.Timedelta(days=1):%d %b %Y}; BTC (L=14) and ETH (L=56) USDⓈ-M perpetuals, "
        "short only, 0.5% risk, 0.05% fees a side, real funding, every portfolio limit.",
        "",
        "| Strategy | Trades | Win | Profit factor | Avg R | Max drawdown | Return | Bootstrap low bound | Funding paid (USDT; - = received) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        _row("C2-short (verdict)", m),
    ]
    if combined is not None:
        lines.append(_row("C2 long (spot) + C2-short together, information only", combined))
    lines += ["", "Per coin (trades, profit factor, total R): " + "; ".join(f"{s} {v}" for s, v in m["per_symbol"].items()),
              f"Fingerprint `{fp}`.", "", "Data: " + "; ".join(notes)]
    return "\n".join(lines) + "\n"


async def dev(data_dir: Path, settings, db, progress: Callable[[str], None] = print) -> tuple[bool, Path]:
    data = load_perps(data_dir)
    fp = fingerprint(settings, data)
    res = run(_short_legs(data, settings.cost_model), settings, DEV_FROM, DEV_END)
    m = summary(res)
    ok, why = judge(m, "dev")
    out = data_dir / "reports" / f"round5-dev-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render("dev", ok, why, m, fp, [d.note for d in data.values()]))
    (out / "result.json").write_text(json.dumps({"verdict": "PASS" if ok else "FAIL", "reasons": why, "metrics": m, "fingerprint": fp}, indent=2, default=str))
    pd.DataFrame([asdict(t) for t in res.trades]).to_csv(out / "trades.csv", index=False)
    async with db.sessions() as s, s.begin():
        row = await s.get(StateRow, DEV_KEY)
        runs = (row.value.get("runs", 0) if row else 0) + 1
        value = {"at": datetime.now(UTC).isoformat(timespec="seconds"), "verdict": "PASS" if ok else "FAIL", "fingerprint": fp,
                 "runs": runs, "report": str(out / "report.md")}
        if row is None:
            s.add(StateRow(key=DEV_KEY, value=value))
        else:
            row.value = value
    return ok, out


async def holdout(data_dir: Path, settings, db, progress: Callable[[str], None] = print) -> tuple[bool, Path]:
    from .round2 import holdout_record

    pl = plan()
    done = await holdout_record(db, STRATEGY, pl)
    if done:
        raise SystemExit(f"Round 5's unseen look was already taken on {done.get('at')} ({done.get('verdict')}); it is locked. "
                         f"Report: {done.get('report')}")
    data = load_perps(data_dir)
    fp = fingerprint(settings, data)
    async with db.sessions() as s:
        d = await s.get(StateRow, DEV_KEY)
    if d is None or d.value.get("verdict") != "PASS":
        raise SystemExit("The unseen look needs a development PASS first (`tradeengine round5 --dev`).")
    if d.value.get("fingerprint") != fp:
        raise SystemExit("The code, data, costs or settings changed since the development PASS: run --dev again first.")
    shorts = _short_legs(data, settings.cost_model)
    res = run(shorts, settings, DEV_END, HOLDOUT_END)
    m = summary(res)
    ok, why = judge(m, "holdout")
    longs, notes = _long_legs(data_dir, settings.cost_model)
    combined = summary(run(longs + shorts, settings, DEV_END, HOLDOUT_END)) if len(longs) == len(PERPS) else None
    out = data_dir / "reports" / f"round5-holdout-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render("holdout", ok, why, m, fp, [x.note for x in data.values()] + notes, combined))
    (out / "result.json").write_text(json.dumps({"verdict": "PASS" if ok else "FAIL", "reasons": why, "metrics": m,
                                                 "combined_information_only": combined, "fingerprint": fp}, indent=2, default=str))
    pd.DataFrame([asdict(t) for t in res.trades]).to_csv(out / "trades.csv", index=False)
    await record_holdout(db, STRATEGY, {"at": datetime.now(UTC).isoformat(timespec="seconds"), "verdict": "PASS" if ok else "FAIL",
                                       "fingerprint": fp, "report": str(out / "report.md")}, pl)
    return ok, out
