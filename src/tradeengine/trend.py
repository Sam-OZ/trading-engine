"""Round 2 trend-following strategies (pre-registered 12 Nov 2026), on daily and 4-hour bars.

Same contract as the round 1 strategies, so the backtester and the live runner share one code path:
    prepare(h1, params, crypto=...) -> frame at the strategy's own timeframe, COMPLETED bars only
    signal_at(cols, i, params, allow_short) -> Signal | None   (rows <= i only)
    exit_at(cols, i, side, params) -> bool                     (signal-based exit at bar i's close)

Every frame has `closes_at` (UTC time the bar closed) and `ready` (all inputs finite). A signal is
formed at the CLOSE of bar i and acted on at the open of the next bar.

Bars are built from hourly bars:
- Daily, Forex and gold: the session closes at 17:00 New York time (the market convention, DST-aware).
  The Monday bar runs Sunday 17:00 to Monday 17:00 New York. Weekend sessions are dropped.
- Daily, crypto: 00:00 to 24:00 UTC, seven days a week.
- 4-hour: UTC-aligned (00:00, 04:00, ...).
A bar is kept only once the hourly data has reached its close, so the live runner never acts on a
half-built day.

The three hypotheses (exact rules and grids from the pre-registered plan):
- H1 channel_breakout_d1: daily close beyond the prior N-day high/low; stop k x ATR(20); ATR trail t from the start.
- H2 tsmom_d1: sign of the L-day return, checked at Friday's daily close; exit when the sign flips;
  safety stop 3 x ATR(20), trailed at 4 x ATR.
- H3 breakout_h4_dfilter: 4-hour close beyond the prior N-bar channel, only in the direction of the
  daily close vs its 200-day EMA; stop k x ATR(14, 4-hour), trailed at 3 x ATR.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import indicators as ind
from .models import Side
from .strategies import Signal

NEW_YORK = ZoneInfo("America/New_York")
_OHLCV = ["open", "high", "low", "close", "volume"]
_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _finite(*xs) -> bool:
    return all(x is not None and math.isfinite(float(x)) for x in xs)


# ---------------------------------------------------------------------------------- bars
def _last_close(h1: pd.DataFrame) -> pd.Timestamp:
    return h1.index[-1] + pd.Timedelta(hours=1)


def daily_from_h1(h1: pd.DataFrame, crypto: bool) -> pd.DataFrame:
    """Daily bars indexed by session OPEN time (UTC); columns OHLCV + closes_at + session_date + weekday."""
    df = h1[_OHLCV].astype(float).sort_index()
    if df.empty:
        return pd.DataFrame(columns=[*_OHLCV, "closes_at", "session_date", "weekday"])
    if crypto:
        key = df.index.floor("D").tz_localize(None)  # UTC calendar day
        out = df.groupby(key).agg(_AGG)
        start = out.index.tz_localize("UTC")
        closes = start + pd.Timedelta(days=1)
        session = out.index
    else:
        ny = df.index.tz_convert(NEW_YORK)
        session_naive = (ny + pd.Timedelta(hours=7)).tz_localize(None).normalize()  # 17:00 NY rolls to the next date
        out = df.groupby(session_naive).agg(_AGG)
        session = out.index
        out = out[session.day_of_week < 5]
        session = out.index
        closes = (session + pd.Timedelta(hours=17)).tz_localize(NEW_YORK).tz_convert("UTC")
        start = (session - pd.Timedelta(days=1) + pd.Timedelta(hours=17)).tz_localize(NEW_YORK).tz_convert("UTC")
    out = out.copy()
    out.index = pd.DatetimeIndex(start, name="open_time")
    out["closes_at"] = np.asarray(closes)
    out["closes_at"] = pd.to_datetime(out["closes_at"], utc=True)
    out["session_date"] = np.asarray(session)
    out["weekday"] = np.asarray(session.day_of_week)
    out = out.dropna(subset=["open", "close"])
    return out[out["closes_at"] <= _last_close(h1)]  # completed sessions only


def h4_bars(h1: pd.DataFrame) -> pd.DataFrame:
    df = h1[_OHLCV].astype(float).sort_index()
    out = df.resample("4h", origin="epoch", label="left", closed="left").agg(_AGG).dropna(subset=["open", "close"])
    out["closes_at"] = out.index + pd.Timedelta(hours=4)
    return out[out["closes_at"] <= _last_close(h1)] if len(df) else out


# ---------------------------------------------------------------------------------- strategies
@dataclass
class ChannelBreakoutD1:
    """H1: daily close beyond the prior N-day channel; stop k x ATR(20); trail t x ATR from the best close."""

    name: str = "channel_breakout_d1"
    timeframe: str = "D1"
    prep_keys: tuple = ("dc_n",)
    min_is_trades: int = 6
    min_history_days: int = 400
    trail_activate_r: float = 0.0
    default_params: dict = field(default_factory=lambda: {"dc_n": 55, "stop_atr": 2.0, "trail_atr": 3.0})
    grid: dict = field(default_factory=lambda: {"dc_n": [55, 100], "stop_atr": [2.0, 3.0], "trail_atr": [3.0, 4.0]})

    def prepare(self, h1: pd.DataFrame, params: dict, crypto: bool = False) -> pd.DataFrame:
        d = daily_from_h1(h1, crypto)
        dc = ind.donchian(d["high"], d["low"], int(params["dc_n"]))
        d["dc_high"], d["dc_low"] = dc["dc_high"], dc["dc_low"]
        d["atr"] = ind.atr(d["high"], d["low"], d["close"], 20)
        d["ready"] = d[["dc_high", "dc_low", "atr"]].notna().all(axis=1)
        return d

    def signal_at(self, cols, i, p, allow_short):
        c, atr, hi, lo = cols["close"][i], cols["atr"][i], cols["dc_high"][i], cols["dc_low"][i]
        if not _finite(c, atr, hi, lo) or atr <= 0:
            return None
        if c > hi:
            return Signal(Side.BUY, c, c - p["stop_atr"] * atr, None, atr, f"daily close above the prior {p['dc_n']}-day high",
                          p["trail_atr"], None, None, 0.0)
        if allow_short and c < lo:
            return Signal(Side.SELL, c, c + p["stop_atr"] * atr, None, atr, f"daily close below the prior {p['dc_n']}-day low",
                          p["trail_atr"], None, None, 0.0)
        return None

    def exit_at(self, cols, i, side, p) -> bool:
        return False


@dataclass
class TimeSeriesMomentumD1:
    """H2: long when the L-day return is positive, short when negative; checked at Friday's daily close."""

    name: str = "tsmom_d1"
    timeframe: str = "D1"
    prep_keys: tuple = ("lookback",)
    min_is_trades: int = 6
    min_history_days: int = 450
    trail_activate_r: float = 0.0
    default_params: dict = field(default_factory=lambda: {"lookback": 126, "stop_atr": 3.0, "trail_atr": 4.0})
    grid: dict = field(default_factory=lambda: {"lookback": [63, 126, 252], "stop_atr": [3.0], "trail_atr": [4.0]})

    def prepare(self, h1: pd.DataFrame, params: dict, crypto: bool = False) -> pd.DataFrame:
        d = daily_from_h1(h1, crypto)
        n = int(params["lookback"])
        d["ret"] = d["close"] / d["close"].shift(n) - 1.0
        d["atr"] = ind.atr(d["high"], d["low"], d["close"], 20)
        d["week_end"] = (d["weekday"] == 4).astype(float)  # Friday's session
        d["ready"] = d[["ret", "atr"]].notna().all(axis=1)
        return d

    def signal_at(self, cols, i, p, allow_short):
        if cols["week_end"][i] != 1.0:
            return None
        c, atr, ret = cols["close"][i], cols["atr"][i], cols["ret"][i]
        if not _finite(c, atr, ret) or atr <= 0:
            return None
        if ret > 0:
            return Signal(Side.BUY, c, c - p["stop_atr"] * atr, None, atr, f"{p['lookback']}-day return positive ({ret:+.1%})",
                          p["trail_atr"], None, None, 0.0)
        if allow_short and ret < 0:
            return Signal(Side.SELL, c, c + p["stop_atr"] * atr, None, atr, f"{p['lookback']}-day return negative ({ret:+.1%})",
                          p["trail_atr"], None, None, 0.0)
        return None

    def exit_at(self, cols, i, side, p) -> bool:
        if cols["week_end"][i] != 1.0 or not _finite(cols["ret"][i]):
            return False
        ret = cols["ret"][i]
        return (side is Side.BUY and ret < 0) or (side is Side.SELL and ret > 0)


@dataclass
class BreakoutH4DailyFilter:
    """H3: 4-hour close beyond the prior N-bar channel, only in the direction of the daily close vs its 200-day EMA."""

    name: str = "breakout_h4_dfilter"
    timeframe: str = "H4"
    prep_keys: tuple = ("dc_n",)
    min_is_trades: int = 12
    min_history_days: int = 450
    trail_activate_r: float = 0.0
    default_params: dict = field(default_factory=lambda: {"dc_n": 20, "stop_atr": 2.0, "trail_atr": 3.0})
    grid: dict = field(default_factory=lambda: {"dc_n": [20, 55], "stop_atr": [2.0, 3.0], "trail_atr": [3.0]})

    def prepare(self, h1: pd.DataFrame, params: dict, crypto: bool = False) -> pd.DataFrame:
        f = h4_bars(h1)
        dc = ind.donchian(f["high"], f["low"], int(params["dc_n"]))
        f["dc_high"], f["dc_low"] = dc["dc_high"], dc["dc_low"]
        f["atr"] = ind.atr(f["high"], f["low"], f["close"], 14)
        d = daily_from_h1(h1, crypto)
        daily = pd.DataFrame({"available_at": d["closes_at"].to_numpy(), "d_close": d["close"].to_numpy(),
                              "d_ema200": ind.ema(d["close"], 200).to_numpy()})
        if len(f) and len(daily):
            merged = pd.merge_asof(
                pd.DataFrame({"closes_at": f["closes_at"].to_numpy()}),
                daily.sort_values("available_at"),
                left_on="closes_at", right_on="available_at", direction="backward",
            )
            f["d_close"], f["d_ema200"] = merged["d_close"].to_numpy(), merged["d_ema200"].to_numpy()
        else:
            f["d_close"], f["d_ema200"] = np.nan, np.nan
        f["ready"] = f[["dc_high", "dc_low", "atr", "d_close", "d_ema200"]].notna().all(axis=1)
        return f

    def signal_at(self, cols, i, p, allow_short):
        c, atr, hi, lo = cols["close"][i], cols["atr"][i], cols["dc_high"][i], cols["dc_low"][i]
        dc, de = cols["d_close"][i], cols["d_ema200"][i]
        if not _finite(c, atr, hi, lo, dc, de) or atr <= 0:
            return None
        if dc > de and c > hi:
            return Signal(Side.BUY, c, c - p["stop_atr"] * atr, None, atr,
                          f"daily close above its 200-day EMA; 4-hour close above the prior {p['dc_n']}-bar high",
                          p["trail_atr"], None, None, 0.0)
        if allow_short and dc < de and c < lo:
            return Signal(Side.SELL, c, c + p["stop_atr"] * atr, None, atr,
                          f"daily close below its 200-day EMA; 4-hour close below the prior {p['dc_n']}-bar low",
                          p["trail_atr"], None, None, 0.0)
        return None

    def exit_at(self, cols, i, side, p) -> bool:
        return False


@dataclass
class TimeSeriesMomentumFastD1:
    """Round 3 C2: hold while the L-day return is positive (short while negative where shorting is
    allowed); checked at every daily close; exit at the next open once it is no longer positive.
    Safety stop 3 x ATR(20), trailed at 4 x ATR."""

    name: str = "tsmom_fast_d1"
    timeframe: str = "D1"
    prep_keys: tuple = ("lookback",)
    min_is_trades: int = 8
    min_history_days: int = 120
    trail_activate_r: float = 0.0
    default_params: dict = field(default_factory=lambda: {"lookback": 28, "stop_atr": 3.0, "trail_atr": 4.0})
    grid: dict = field(default_factory=lambda: {"lookback": [14, 28, 56], "stop_atr": [3.0], "trail_atr": [4.0]})

    def prepare(self, h1: pd.DataFrame, params: dict, crypto: bool = False) -> pd.DataFrame:
        d = daily_from_h1(h1, crypto)
        n = int(params["lookback"])
        d["ret"] = d["close"] / d["close"].shift(n) - 1.0
        d["atr"] = ind.atr(d["high"], d["low"], d["close"], 20)
        d["ready"] = d[["ret", "atr"]].notna().all(axis=1)
        return d

    def signal_at(self, cols, i, p, allow_short):
        c, atr, ret = cols["close"][i], cols["atr"][i], cols["ret"][i]
        if not _finite(c, atr, ret) or atr <= 0:
            return None
        if ret > 0:
            return Signal(Side.BUY, c, c - p["stop_atr"] * atr, None, atr, f"{p['lookback']}-day return positive ({ret:+.1%})",
                          p["trail_atr"], None, None, 0.0)
        if allow_short and ret < 0:
            return Signal(Side.SELL, c, c + p["stop_atr"] * atr, None, atr, f"{p['lookback']}-day return negative ({ret:+.1%})",
                          p["trail_atr"], None, None, 0.0)
        return None

    def exit_at(self, cols, i, side, p) -> bool:
        ret = cols["ret"][i]
        if not _finite(ret):
            return False
        return (side is Side.BUY and ret <= 0) or (side is Side.SELL and ret >= 0)


ROUND2_STRATEGIES = {s.name: s for s in (ChannelBreakoutD1(), TimeSeriesMomentumD1(), BreakoutH4DailyFilter())}
# Round 3 (crypto, pre-registered 12 Nov 2026): C1 is round 2's 4-hour rule unchanged; C2 is new.
ROUND3_STRATEGIES = {s.name: s for s in (BreakoutH4DailyFilter(), TimeSeriesMomentumFastD1())}


def prep_key(strategy, params: dict) -> str:
    keys = getattr(strategy, "prep_keys", ("bb_std", "dc_n"))
    return ",".join(f"{k}={params[k]}" for k in keys if k in params)


def prepare(strategy, h1: pd.DataFrame, params: dict, crypto: bool) -> pd.DataFrame:
    """Call prepare() the right way for round 1 (H1, no crypto flag) and round 2 strategies."""
    if getattr(strategy, "timeframe", "H1") == "H1":
        return strategy.prepare(h1, params)
    return strategy.prepare(h1, params, crypto=crypto)
