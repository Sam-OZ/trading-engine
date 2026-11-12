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


async def download_dukascopy(
    client: httpx.AsyncClient, symbol: str, start: date, end: date, concurrency: int = 8, retries: int = 3
) -> list[Bar]:
    """symbol like "EUR/USD". BID minute candles for each day in [start, end), fetched in parallel."""
    import asyncio

    code = symbol.replace("/", "").upper()
    days = [start + timedelta(days=k) for k in range((end - start).days)]
    sem = asyncio.Semaphore(concurrency)

    async def one(d: date) -> list[Bar]:
        url = f"{DUKASCOPY_URL}/{code}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
        for attempt in range(retries):
            async with sem:
                try:
                    r = await client.get(url)
                except httpx.TransportError:
                    await asyncio.sleep(1 + attempt)
                    continue
            if r.status_code == 200:
                return decode_dukascopy_candles(r.content, d, symbol)
            if r.status_code == 404:
                return []
            await asyncio.sleep(1 + attempt)
        raise RuntimeError(f"Dukascopy {symbol} {d}: failed after {retries} attempts")

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


async def download_dukascopy_h1(
    client: httpx.AsyncClient, symbol: str, start: date, end: date, concurrency: int = 8, retries: int = 3, today: date | None = None,
) -> tuple[list[Bar], dict[str, int]]:
    """Hourly BID candles for [start, end), one request per month where Dukascopy has a monthly
    file of hourly candles; otherwise (404, or a file that fails the checks) the month is built from
    its minute files. The current month always comes from minute files (its monthly file is partial).
    Returns (bars, {"monthly": n, "from_minutes": n}).
    """
    import asyncio

    code = symbol.replace("/", "").upper()
    today = today or datetime.now(UTC).date()
    sem = asyncio.Semaphore(concurrency)
    stats = {"monthly": 0, "from_minutes": 0}

    async def get(url: str) -> httpx.Response | None:
        for attempt in range(retries):
            async with sem:
                try:
                    r = await client.get(url)
                except httpx.TransportError:
                    await asyncio.sleep(1 + attempt)
                    continue
            if r.status_code in (200, 404):
                return r
            await asyncio.sleep(1 + attempt)
        raise RuntimeError(f"Dukascopy {url}: failed after {retries} attempts")

    msem = asyncio.Semaphore(4)  # months in flight; a month built from minute files runs its own 8 requests

    async def month(m0: date) -> list[Bar]:
        async with msem:
            return await _month(m0)

    async def _month(m0: date) -> list[Bar]:
        m1 = date(m0.year + (m0.month == 12), m0.month % 12 + 1, 1)
        current = m0 <= today < m1
        if not current:
            r = await get(f"{DUKASCOPY_URL}/{code}/{m0.year}/{m0.month - 1:02d}/BID_candles_hour_1.bi5")
            if r is not None and r.status_code == 200:
                base = datetime(m0.year, m0.month, 1, tzinfo=UTC)
                try:
                    bars = decode_candles(r.content, base, symbol, "H1")
                    ok = all(b.open_time.minute == 0 and b.open_time.second == 0 and b.open_time < datetime(m1.year, m1.month, 1, tzinfo=UTC)
                             for b in bars)
                except (ValueError, lzma.LZMAError):
                    ok = False
                if ok:
                    stats["monthly"] += 1
                    return bars
        stats["from_minutes"] += 1
        last = min(m1, end, today + timedelta(days=1))
        first = max(m0, start)
        minutes = await download_dukascopy(client, symbol, first, last, concurrency=concurrency, retries=retries) if last > first else []
        if not minutes:
            return []
        return frame_to_bars(resample_frame(bars_to_frame(minutes), "1h"), symbol, "H1")

    out: list[Bar] = []
    for bars in await asyncio.gather(*(month(m) for m in _month_starts(start, end))):
        out.extend(b for b in bars if start <= b.open_time.date() < end)
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
