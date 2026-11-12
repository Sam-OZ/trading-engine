"""Historical data download and storage for backtesting (Phase 2 uses these files).

Sources (all free):
- Binance public klines (real exchange data) via the live public REST endpoint.
- Capital.com price history via the demo API (max 1,000 bars per request).
- Dukascopy minute candles for FX (10+ years): one LZMA-compressed .bi5 file per
  day, 24-byte big-endian records: seconds offset, open, close, low, high
  (uint32, scaled by the instrument's point) and volume (float32).

Files: data/history/<source>/<SYMBOL>_<TF>.csv.gz, one row per bar, merged on re-download.
"""

from __future__ import annotations

import csv
import gzip
import lzma
import struct
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import httpx
import pandas as pd

from .models import Bar

DUKASCOPY_URL = "https://datafeed.dukascopy.com/datafeed"
_REC = struct.Struct(">IIIIIf")


def dukascopy_point(symbol: str) -> int:
    """Price scale: JPY-quoted pairs and gold use 3 decimals, other FX pairs 5.

    A wrong scale is caught by `check_price_range` (prices 10x or 100x off), not silently used.
    """
    s = symbol.upper().replace("/", "")
    return 1000 if s.endswith("JPY") or s.startswith("XAU") else 100000


# Loose sanity ranges for 2010-2030 prices: a wrong decimal scale lands 10x or 100x outside them.
PRICE_RANGES: dict[str, tuple[float, float]] = {
    "EUR/USD": (0.8, 1.7), "GBP/USD": (1.0, 2.2), "USD/JPY": (70, 200), "AUD/USD": (0.5, 1.2), "USD/CAD": (0.9, 1.7),
    "USD/CHF": (0.6, 1.3), "NZD/USD": (0.45, 1.0), "EUR/GBP": (0.65, 1.0), "EUR/JPY": (90, 220), "GBP/JPY": (110, 260),
    "XAU/USD": (900, 9000),
}


def check_price_range(symbol: str, bars: list[Bar]) -> None:
    rng = PRICE_RANGES.get(symbol)
    if rng is None or not bars:
        return
    closes = sorted(float(b.close) for b in bars)
    median = closes[len(closes) // 2]
    if not (rng[0] <= median <= rng[1]):
        raise ValueError(f"{symbol}: median price {median:g} outside {rng[0]:g}-{rng[1]:g}; the Dukascopy price scale is wrong")


def decode_dukascopy_candles(raw: bytes, day: date, symbol: str) -> list[Bar]:
    """One day of minute candles."""
    return decode_candles(raw, datetime(day.year, day.month, day.day, tzinfo=UTC), symbol, "M1")


def decode_candles(raw: bytes, base: datetime, symbol: str, timeframe: str) -> list[Bar]:
    """Dukascopy candle file: records are (seconds from `base`, open, close, low, high, volume)."""
    if not raw:
        return []  # weekend / holiday
    data = lzma.decompress(raw)
    if len(data) % _REC.size:
        raise ValueError(f"Dukascopy record size mismatch for {symbol} {base:%Y-%m-%d}")
    scale = Decimal(dukascopy_point(symbol))
    bars = []
    for off, o, c, lo, hi, vol in _REC.iter_unpack(data):
        if vol == 0 and o == c == lo == hi:
            continue  # Dukascopy pads quiet periods with flat zero-volume candles
        bar = Bar(
            symbol=symbol,
            timeframe=timeframe,
            open_time=base + timedelta(seconds=off),
            open=Decimal(o) / scale,
            high=Decimal(hi) / scale,
            low=Decimal(lo) / scale,
            close=Decimal(c) / scale,
            volume=Decimal(str(round(vol, 6))),
        )
        if not (bar.low <= min(bar.open, bar.close) and max(bar.open, bar.close) <= bar.high):
            raise ValueError(f"Dukascopy candle failed sanity check at {bar.open_time}: field order changed?")
        bars.append(bar)
    return bars


class Throttle:
    """Polite pacing for one data host: at most `concurrency` requests in flight and at least
    `interval` seconds between request starts. Dukascopy answers HTTP 429 when asked too fast."""

    def __init__(self, concurrency: int = 2, interval: float = 0.4):
        import asyncio

        self.sem = asyncio.Semaphore(concurrency)
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def __aenter__(self):
        import asyncio
        import time

        await self.sem.acquire()
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *exc):
        self.sem.release()

    async def pause(self, seconds: float) -> None:
        """After a 429: hold back every request, not just the one that was refused."""
        import time

        async with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


async def _fetch(client: httpx.AsyncClient, url: str, throttle, retries: int = 4, busy_retries: int = 8) -> httpx.Response:
    """GET through the throttle. Returns a 200 or 404 response; otherwise raises with what the
    server last said. HTTP 429 ("too many requests") waits Retry-After or 15, 30, 60, 120 s
    (capped at 120) and gets more attempts than other errors (1, 2, 4, 8 s)."""
    import asyncio

    last, busy, other = "no response", 0, 0
    while True:
        async with throttle:
            try:
                r = await client.get(url)
            except httpx.TransportError as e:
                r, last = None, type(e).__name__
        if r is not None and r.status_code in (200, 404):
            return r
        if r is not None and r.status_code == 429:
            busy += 1
            last = "HTTP 429"
            if busy >= busy_retries:
                break
            try:
                wait = float(r.headers.get("Retry-After", ""))
            except ValueError:
                wait = min(120.0, 15.0 * 2 ** (busy - 1))
            await throttle.pause(wait)
            continue
        if r is not None:
            last = f"HTTP {r.status_code}"
        other += 1
        if other >= retries:
            break
        await asyncio.sleep(2 ** (other - 1))
    raise RuntimeError(f"Dukascopy {url}: {last} after {busy + other} attempts")


async def download_dukascopy(
    client: httpx.AsyncClient, symbol: str, start: date, end: date, concurrency: int = 2, retries: int = 4, throttle=None,
) -> list[Bar]:
    """symbol like "EUR/USD". BID minute candles for each day in [start, end)."""
    import asyncio

    code = symbol.replace("/", "").upper()
    days = [start + timedelta(days=k) for k in range((end - start).days)]
    throttle = throttle or Throttle(concurrency)

    async def one(d: date) -> list[Bar]:
        r = await _fetch(client, f"{DUKASCOPY_URL}/{code}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5", throttle, retries)
        return decode_dukascopy_candles(r.content, d, symbol) if r.status_code == 200 else []

    out: list[Bar] = []
    for bars in await asyncio.gather(*(one(d) for d in days)):
        out.extend(bars)
    return out


def _month_starts(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d < end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def complete_months(df: pd.DataFrame | None) -> set[tuple[int, int]]:
    """(year, month) pairs the file already covers end to end: bars within the first and last 4 days
    of the month and at least 300 hourly bars. A month the file only partly covers is not counted."""
    if df is None or df.empty:
        return set()
    out = set()
    idx = df.index
    for (y, m), g in pd.Series(1, index=idx).groupby([idx.year, idx.month]):
        days_in_month = pd.Period(f"{y}-{m:02d}").days_in_month
        first, last = g.index[0].day, g.index[-1].day
        if first <= 4 and last >= days_in_month - 3 and len(g) >= 300:
            out.add((int(y), int(m)))
    return out


async def download_dukascopy_h1(
    client: httpx.AsyncClient, symbol: str, start: date, end: date, today: date | None = None,
    have: set[tuple[int, int]] | None = None, throttle=None, progress=None,
) -> tuple[list[Bar], dict]:
    """Hourly BID candles for [start, end), one request per month: Dukascopy's monthly file of hourly
    candles. A month whose monthly file does not exist (404) or fails the checks is built from its
    daily minute files; the current month always is (its monthly file is partial).

    Resumable: months in `have` (already complete on disk) are skipped, except the latest one, which
    is fetched again so the new data can be checked against the file. A month that still fails after
    the retries (e.g. the server keeps answering 429) is reported in stats["failed"], and the months
    that did arrive are returned, so a re-run continues where this one stopped.
    Returns (bars, {"monthly", "from_minutes", "skipped", "failed": ["YYYY-MM: reason", ...]}).
    """
    import asyncio

    code = symbol.replace("/", "").upper()
    today = today or datetime.now(UTC).date()
    throttle = throttle or Throttle()
    have = set(have or ())
    verify = max(have) if have else None
    stats: dict = {"monthly": 0, "from_minutes": 0, "skipped": 0, "failed": []}

    async def month(m0: date) -> list[Bar]:
        m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
        current = m0 <= today < m1
        if not current:
            r = await _fetch(client, f"{DUKASCOPY_URL}/{code}/{m0.year}/{m0.month - 1:02d}/BID_candles_hour_1.bi5", throttle)
            if r.status_code == 200:
                try:
                    bars = decode_candles(r.content, datetime(m0.year, m0.month, 1, tzinfo=UTC), symbol, "H1")
                    end_of_month = datetime(m1.year, m1.month, 1, tzinfo=UTC)
                    if all(b.open_time.minute == 0 and b.open_time.second == 0 and b.open_time < end_of_month for b in bars):
                        stats["monthly"] += 1
                        return bars
                except (ValueError, lzma.LZMAError):
                    pass
        stats["from_minutes"] += 1
        first, last = max(m0, start), min(m1, end, today + timedelta(days=1))
        minutes = await download_dukascopy(client, symbol, first, last, throttle=throttle) if last > first else []
        return frame_to_bars(resample_frame(bars_to_frame(minutes), "1h"), symbol, "H1") if minutes else []

    async def guarded(m0: date) -> list[Bar]:
        try:
            bars = await month(m0)
        except RuntimeError as e:
            stats["failed"].append(f"{m0:%Y-%m}: {str(e).split(': ', 1)[-1]}")
            return []
        if progress is not None and m0.month == 12:
            progress(f"{m0:%Y} done")
        return bars

    todo = []
    for m0 in _month_starts(start, end):
        key = (m0.year, m0.month)
        if key in have and key != verify and not (m0 <= today < date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)):
            stats["skipped"] += 1
            continue
        todo.append(m0)
    out: list[Bar] = []
    for bars in await asyncio.gather(*(guarded(m) for m in todo)):
        out.extend(b for b in bars if start <= b.open_time.date() < end)
    stats["failed"].sort()
    check_price_range(symbol, out)
    return out, stats


def bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [float(b.open) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "close": [float(b.close) for b in bars],
            "volume": [float(b.volume) for b in bars],
        },
        index=pd.DatetimeIndex([b.open_time for b in bars], name="open_time"),
    )
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _d(x) -> Decimal:
    return Decimal(repr(float(x)))


def frame_to_bars(df: pd.DataFrame, symbol: str, timeframe: str) -> list[Bar]:
    return [
        Bar(symbol, timeframe, t.to_pydatetime(), _d(o), _d(h), _d(lo), _d(c), _d(v))
        for t, o, h, lo, c, v in zip(df.index, df["open"], df["high"], df["low"], df["close"], df["volume"], strict=True)
    ]


def resample_frame(df: pd.DataFrame, rule: str = "1h") -> pd.DataFrame:
    out = df.resample(rule, origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "close"])


def read_frame(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, compression="gzip")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.set_index("open_time").astype(float).sort_index()


def load_h1(data_dir: Path, symbol: str, crypto: bool) -> tuple[pd.DataFrame | None, str]:
    """Best available H1 history for a symbol, and a one-line note on where it came from."""
    if crypto:
        candidates = [("binance", history_path(data_dir, "binance", symbol, "H1"))]
    else:
        candidates = [("capital", history_path(data_dir, "capital", symbol, "H1")), ("dukascopy", history_path(data_dir, "dukascopy", symbol, "H1"))]
    best, note = None, f"{symbol}: no history file found (run download-all)"
    for source, path in candidates:
        df = read_frame(path)
        if df is None or df.empty:
            continue
        span = (df.index[-1] - df.index[0]).days
        if best is None or span > (best.index[-1] - best.index[0]).days:
            best = df
            note = f"{symbol}: {source} H1, {len(df):,} bars, {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({span} days)"
    return best, note


def history_path(data_dir: Path, source: str, symbol: str, timeframe: str) -> Path:
    return data_dir / "history" / source / f"{symbol.replace('/', '')}_{timeframe}.csv.gz"


def read_bars(path: Path, symbol: str, timeframe: str) -> list[Bar]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", newline="") as f:
        return [
            Bar(
                symbol,
                timeframe,
                datetime.fromisoformat(r["open_time"]),
                Decimal(r["open"]),
                Decimal(r["high"]),
                Decimal(r["low"]),
                Decimal(r["close"]),
                Decimal(r["volume"]),
            )
            for r in csv.DictReader(f)
        ]


def write_bars(path: Path, bars: list[Bar]) -> int:
    """Merge `bars` into the file (new values win for the same timestamp). Returns row count."""
    if not bars:
        return 0
    existing = {b.open_time: b for b in read_bars(path, bars[0].symbol, bars[0].timeframe)}
    existing.update({b.open_time: b for b in bars})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for t in sorted(existing):
            b = existing[t]
            w.writerow([t.isoformat(), b.open, b.high, b.low, b.close, b.volume])
    tmp.replace(path)
    return len(existing)
