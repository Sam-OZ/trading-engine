"""The three rule-based strategies, written once and used by both the backtester and the live engine.

Shape of every strategy:
    prepare(h1, params)            -> DataFrame of H1 bars + indicator columns (H4 context aligned without look-ahead)
    signal_at(cols, i, params, allow_short) -> Signal | None, using rows <= i only

The live runner calls prepare() on the latest completed bars and asks signal_at() about
the last row; the backtester calls prepare() once and walks i forward. Same functions,
same numbers. A parity test asserts that a signal at row i is identical whether the
frame was prepared on the full history or only on data up to row i.

The signal is formed at the CLOSE of H1 bar i; the order goes in at the open of bar i+1.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from . import indicators as ind
from .models import Side

H1 = pd.Timedelta(hours=1)
H4 = pd.Timedelta(hours=4)
WARMUP_BARS = 4 * 210  # H4 EMA200 needs ~200 H4 bars of history


@dataclass(frozen=True)
class Signal:
    side: Side
    entry: float  # reference price: close of the signal bar
    stop: float
    target: float | None
    atr: float
    reason: str
    trail_atr: float | None = None  # trail by this many ATR once the trade is +1R
    breakeven_at_r: float | None = None  # move stop to entry at this R
    max_bars: int = 72  # time exit (H1 bars)


@dataclass(frozen=True)
class Context:
    """Regime thresholds shared by all strategies (admin-tunable later if needed)."""

    adx_trend: float = 25.0
    adx_range: float = 20.0


# ---------------------------------------------------------------------------------- frame
def h4_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """UTC-aligned 4-hour bars (00:00, 04:00, ...). Indexed by bar open time."""
    agg = h1.resample("4h", origin="epoch", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open", "close"])


def build_frame(h1: pd.DataFrame, bb_std: float = 2.0, dc_n: int = 20, ctx: Context = Context()) -> pd.DataFrame:
    """H1 bars (index = UTC open time; columns open/high/low/close/volume) -> indicator frame."""
    df = h1[["open", "high", "low", "close", "volume"]].astype(float).sort_index().copy()
    c, h, lo = df["close"], df["high"], df["low"]
    df["ema20"] = ind.ema(c, 20)
    df["rsi"] = ind.rsi(c, 14)
    df["atr"] = ind.atr(h, lo, c, 14)
    df["adx"] = ind.adx(h, lo, c, 14)["adx"]
    df["macd_hist"] = ind.macd(c)["macd_hist"]
    bb = ind.bollinger(c, 20, bb_std)
    df[["bb_mid", "bb_upper", "bb_lower", "bb_width"]] = bb[["bb_mid", "bb_upper", "bb_lower", "bb_width"]]
    df["bbw_pct"] = ind.pct_rank(df["bb_width"], 100)
    df["bbw_min10"] = df["bbw_pct"].rolling(10, min_periods=10).min()
    dc = ind.donchian(h, lo, dc_n)
    df["dc_high"], df["dc_low"] = dc["dc_high"], dc["dc_low"]
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=20).mean()
    df["swing_low"] = lo.rolling(5, min_periods=5).min()
    df["swing_high"] = h.rolling(5, min_periods=5).max()

    # Higher-timeframe context: a 4-hour bar is only usable once it has CLOSED.
    h4 = h4_from_h1(df)
    h4i = pd.DataFrame(index=h4.index)
    h4i["h4_ema50"] = ind.ema(h4["close"], 50)
    h4i["h4_ema200"] = ind.ema(h4["close"], 200)
    h4i["h4_adx"] = ind.adx(h4["high"], h4["low"], h4["close"], 14)["adx"]
    h4i["available_at"] = h4i.index + H4
    left = pd.DataFrame({"closes_at": df.index + H1}, index=df.index)
    merged = pd.merge_asof(
        left.reset_index().sort_values("closes_at"),
        h4i.reset_index(drop=True).sort_values("available_at"),
        left_on="closes_at",
        right_on="available_at",
        direction="backward",
    ).set_index(df.index.name or "index")
    for col in ("h4_ema50", "h4_ema200", "h4_adx"):
        df[col] = merged[col].to_numpy()

    up = (df["h4_ema50"] > df["h4_ema200"]) & (df["h4_adx"] >= ctx.adx_trend)
    down = (df["h4_ema50"] < df["h4_ema200"]) & (df["h4_adx"] >= ctx.adx_trend)
    rng = (df["adx"] < ctx.adx_range) & (df["h4_adx"] < ctx.adx_trend)
    df["regime"] = np.select([up, down, rng], [1, -1, 0], default=9)  # 1 up, -1 down, 0 range, 9 neutral
    return df


def columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: df[c].to_numpy() for c in df.columns}


def _finite(*xs: float) -> bool:
    return all(x is not None and math.isfinite(x) for x in xs)


# ---------------------------------------------------------------------------------- strategies
class Strategy(Protocol):
    name: str
    default_params: dict
    grid: dict[str, list]

    def prepare(self, h1: pd.DataFrame, params: dict) -> pd.DataFrame: ...
    def signal_at(self, cols: dict[str, np.ndarray], i: int, params: dict, allow_short: bool) -> Signal | None: ...


def param_grid(grid: dict[str, list]) -> list[dict]:
    keys = sorted(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[k] for k in keys))]


@dataclass
class TrendPullback:
    """Trending market on 4h; buy a pullback to the 1h EMA20 as RSI turns up and MACD momentum improves."""

    name: str = "trend_pullback"
    default_params: dict = field(default_factory=lambda: {"stop_atr": 1.5, "target_r": 2.0, "rsi_pull": 45, "trail_atr": 2.0, "max_bars": 72})
    grid: dict = field(default_factory=lambda: {"stop_atr": [1.0, 1.5, 2.0], "target_r": [2.0, 3.0], "rsi_pull": [40, 45], "trail_atr": [2.0], "max_bars": [72]})

    def prepare(self, h1: pd.DataFrame, params: dict) -> pd.DataFrame:
        return build_frame(h1)

    def signal_at(self, cols, i, p, allow_short):
        if i < 2:
            return None
        c, lo, hi, atr, e20 = cols["close"][i], cols["low"][i], cols["high"][i], cols["atr"][i], cols["ema20"][i]
        r0, r1, r2 = cols["rsi"][i], cols["rsi"][i - 1], cols["rsi"][i - 2]
        m0, m1 = cols["macd_hist"][i], cols["macd_hist"][i - 1]
        regime = cols["regime"][i]
        if not _finite(c, atr, e20, r0, r1, r2, m0, m1) or atr <= 0:
            return None
        if regime == 1 and lo <= e20 + 0.1 * atr and c > e20 and r0 > r1 and min(r1, r2) < p["rsi_pull"] and m0 > m1:
            stop = min(cols["swing_low"][i] - 0.2 * atr, c - p["stop_atr"] * atr)
            risk = c - stop
            return Signal(Side.BUY, c, stop, c + p["target_r"] * risk, atr, "4h uptrend, 1h pullback to EMA20, RSI and MACD turning up",
                          p["trail_atr"], 1.0, p["max_bars"])
        if allow_short and regime == -1 and hi >= e20 - 0.1 * atr and c < e20 and r0 < r1 and max(r1, r2) > 100 - p["rsi_pull"] and m0 < m1:
            stop = max(cols["swing_high"][i] + 0.2 * atr, c + p["stop_atr"] * atr)
            risk = stop - c
            return Signal(Side.SELL, c, stop, c - p["target_r"] * risk, atr, "4h downtrend, 1h rally to EMA20, RSI and MACD turning down",
                          p["trail_atr"], 1.0, p["max_bars"])
        return None


@dataclass
class MeanReversion:
    """Ranging market (low ADX on 1h and 4h); fade a close outside the Bollinger Band with RSI stretched."""

    name: str = "mean_reversion"
    default_params: dict = field(default_factory=lambda: {"bb_std": 2.0, "rsi_low": 30, "stop_atr": 1.5, "max_bars": 24})
    grid: dict = field(default_factory=lambda: {"bb_std": [2.0, 2.5], "rsi_low": [25, 30], "stop_atr": [1.5, 2.0], "max_bars": [24]})

    def prepare(self, h1: pd.DataFrame, params: dict) -> pd.DataFrame:
        return build_frame(h1, bb_std=params["bb_std"])

    def signal_at(self, cols, i, p, allow_short):
        c, atr, rsi, mid = cols["close"][i], cols["atr"][i], cols["rsi"][i], cols["bb_mid"][i]
        lower, upper = cols["bb_lower"][i], cols["bb_upper"][i]
        if cols["regime"][i] != 0 or not _finite(c, atr, rsi, mid, lower, upper) or atr <= 0:
            return None
        if c < lower and rsi < p["rsi_low"]:
            stop = c - p["stop_atr"] * atr
            if mid - c < 0.5 * (c - stop):
                return None
            return Signal(Side.BUY, c, stop, mid, atr, "range: close below lower band, RSI oversold", None, None, p["max_bars"])
        if allow_short and c > upper and rsi > 100 - p["rsi_low"]:
            stop = c + p["stop_atr"] * atr
            if c - mid < 0.5 * (stop - c):
                return None
            return Signal(Side.SELL, c, stop, mid, atr, "range: close above upper band, RSI overbought", None, None, p["max_bars"])
        return None


@dataclass
class VolatilityBreakout:
    """After a Bollinger squeeze, trade a close beyond the prior N-bar Donchian channel on rising volume."""

    name: str = "breakout"
    default_params: dict = field(default_factory=lambda: {"dc_n": 20, "stop_atr": 1.5, "target_r": 4.0, "trail_atr": 2.0, "squeeze_pct": 0.2, "vol_mult": 1.2, "max_bars": 96})
    grid: dict = field(default_factory=lambda: {"dc_n": [20, 40], "stop_atr": [1.5, 2.0], "target_r": [4.0], "trail_atr": [2.0, 3.0], "squeeze_pct": [0.2], "vol_mult": [1.2], "max_bars": [96]})

    def prepare(self, h1: pd.DataFrame, params: dict) -> pd.DataFrame:
        return build_frame(h1, dc_n=params["dc_n"])

    def signal_at(self, cols, i, p, allow_short):
        c, atr = cols["close"][i], cols["atr"][i]
        dch, dcl, squeeze = cols["dc_high"][i], cols["dc_low"][i], cols["bbw_min10"][i]
        vol, vsma = cols["volume"][i], cols["vol_sma20"][i]
        if not _finite(c, atr, dch, dcl, squeeze) or atr <= 0 or squeeze > p["squeeze_pct"]:
            return None
        vol_ok = not (_finite(vsma) and vsma > 0) or vol > p["vol_mult"] * vsma
        if not vol_ok:
            return None
        regime = cols["regime"][i]
        if c > dch and regime != -1:
            stop = c - p["stop_atr"] * atr
            return Signal(Side.BUY, c, stop, c + p["target_r"] * (c - stop), atr, "squeeze breakout above prior channel high",
                          p["trail_atr"], 1.0, p["max_bars"])
        if allow_short and c < dcl and regime != 1:
            stop = c + p["stop_atr"] * atr
            return Signal(Side.SELL, c, stop, c - p["target_r"] * (stop - c), atr, "squeeze breakdown below prior channel low",
                          p["trail_atr"], 1.0, p["max_bars"])
        return None


STRATEGIES: dict[str, Strategy] = {s.name: s for s in (TrendPullback(), MeanReversion(), VolatilityBreakout())}


# ---------------------------------------------------------------------------------- trade management
def trailed_stop(
    side: Side, entry: float, stop: float, initial_risk: float, best_close: float, atr: float,
    trail_atr: float | None, breakeven_at_r: float | None,
) -> float:
    """New stop after a bar closes. Only ever tightens. Shared by backtester and live runner."""
    new = stop
    if not _finite(atr) or initial_risk <= 0:
        return new
    gain = (best_close - entry) if side is Side.BUY else (entry - best_close)
    if breakeven_at_r is not None and gain >= breakeven_at_r * initial_risk:
        new = max(new, entry) if side is Side.BUY else min(new, entry)
    if trail_atr is not None and gain >= initial_risk:
        trail = best_close - trail_atr * atr if side is Side.BUY else best_close + trail_atr * atr
        new = max(new, trail) if side is Side.BUY else min(new, trail)
    return new
