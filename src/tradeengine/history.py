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
    """Price scale: JPY-quoted pairs use 3 decimals, other FX pairs 5."""
    return 1000 if symbol.upper().endswith("JPY") else 100000


def decode_dukascopy_candles(raw: bytes, day: date, symbol: str) -> list[Bar]:
    if not raw:
        return []  # weekend / holiday
    data = lzma.decompress(raw)
    if len(data) % _REC.size:
        raise ValueError(f"Dukascopy record size mismatch for {symbol} {day}")
    scale = Decimal(dukascopy_point(symbol))
    base = datetime(day.year, day.month, day.day, tzinfo=UTC)
    bars = []
    for off, o, c, lo, hi, vol in _REC.iter_unpack(data):
        if vol == 0 and o == c == lo == hi:
            continue  # Dukascopy pads quiet minutes with flat zero-volume candles
        bar = Bar(
            symbol=symbol,
            timeframe="M1",
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
