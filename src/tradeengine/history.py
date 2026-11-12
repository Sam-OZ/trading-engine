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


async def download_dukascopy(client: httpx.AsyncClient, symbol: str, start: date, end: date) -> list[Bar]:
    """symbol like "EUR/USD". Downloads BID minute candles for each day in [start, end)."""
    code = symbol.replace("/", "").upper()
    out: list[Bar] = []
    d = start
    while d < end:
        url = f"{DUKASCOPY_URL}/{code}/{d.year}/{d.month - 1:02d}/{d.day:02d}/BID_candles_min_1.bi5"
        r = await client.get(url)
        if r.status_code == 200:
            out.extend(decode_dukascopy_candles(r.content, d, symbol))
        elif r.status_code != 404:
            r.raise_for_status()
        d += timedelta(days=1)
    return out


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
