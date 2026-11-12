"""Technical indicators on pandas Series/DataFrames (float64), TA-Lib conventions.

Values match TA-Lib (verified against TA-Lib 0.6 to ~1e-12 on random data;
reference values are frozen in tests/test_indicators.py so the tests do not
need TA-Lib installed). Every value at row i uses rows <= i only (no look-ahead); `donchian`
excludes the current bar on purpose so "close breaks the prior N-bar high" is
expressible without peeking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    """EMA seeded with the SMA of the first n values (TA-Lib style)."""
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan)
    alpha = 2.0 / (n + 1)
    valid = np.flatnonzero(~np.isnan(values))
    if len(valid) < n:
        return pd.Series(out, index=s.index)
    start = valid[0]
    seed_end = start + n - 1
    out[seed_end] = values[start : seed_end + 1].mean()
    for i in range(seed_end + 1, len(values)):
        v = values[i]
        out[i] = out[i - 1] if np.isnan(v) else alpha * v + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=s.index)


def _wilder(values: np.ndarray, n: int, first: int) -> np.ndarray:
    """Wilder's running average: seed = mean of values[first : first+n], then (prev*(n-1)+v)/n."""
    out = np.full(len(values), np.nan)
    if len(values) - first < n:
        return out
    seed = first + n - 1
    out[seed] = values[first : seed + 1].mean()
    for i in range(seed + 1, len(values)):
        out[i] = (out[i - 1] * (n - 1) + values[i]) / n
    return out


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = close.to_numpy(dtype=float)
    delta = np.diff(c, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder(gain, n, 1)
    avg_loss = _wilder(loss, n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    out = np.where((avg_loss == 0) & ~np.isnan(avg_gain), 100.0, out)
    return pd.Series(out, index=close.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close).to_numpy(dtype=float)
    return pd.Series(_wilder(tr, n, 1), index=close.index)


def _ta_smooth(values: np.ndarray, n: int) -> np.ndarray:
    """TA-Lib's Wilder smoothing for DM/TR sums: start with the sum of bars 1..n-1,
    then S_i = S_(i-1) - S_(i-1)/n + x_i from bar n on (first output at index n)."""
    out = np.full(len(values), np.nan)
    if len(values) <= n:
        return out
    s = values[1:n].sum()
    for i in range(n, len(values)):
        s = s - s / n + values[i]
        out[i] = s
    return out


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.DataFrame:
    """Returns columns plus_di, minus_di, adx, matching TA-Lib's PLUS_DI / MINUS_DI / ADX."""
    h, lo = high.to_numpy(dtype=float), low.to_numpy(dtype=float)
    up = np.diff(h, prepend=np.nan)
    down = -np.diff(lo, prepend=np.nan)
    plus_dm = np.where((up > 0) & (up > down), up, 0.0)
    minus_dm = np.where((down > 0) & (down > up), down, 0.0)
    tr = true_range(high, low, close).to_numpy(dtype=float)
    tr_s, pdm_s, mdm_s = _ta_smooth(tr, n), _ta_smooth(plus_dm, n), _ta_smooth(minus_dm, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(tr_s > 0, 100.0 * pdm_s / tr_s, 0.0)
        mdi = np.where(tr_s > 0, 100.0 * mdm_s / tr_s, 0.0)
        dx = np.where(pdi + mdi > 0, 100.0 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
    pdi[np.isnan(tr_s)] = np.nan
    mdi[np.isnan(tr_s)] = np.nan
    adx_v = _wilder(dx, n, n) if len(dx) >= 2 * n else np.full(len(dx), np.nan)
    return pd.DataFrame({"plus_di": pdi, "minus_di": mdi, "adx": adx_v}, index=close.index)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """TA-Lib alignment: the fast EMA is seeded over the same window end as the slow one,
    so both lines start on bar slow-1 and the signal line on bar slow+signal-2."""
    c = close.to_numpy(dtype=float)
    out_fast = np.full(len(c), np.nan)
    out_slow = np.full(len(c), np.nan)
    if len(c) >= slow:
        for out, n in ((out_fast, fast), (out_slow, slow)):
            a = 2.0 / (n + 1)
            out[slow - 1] = c[slow - n : slow].mean()
            for i in range(slow, len(c)):
                out[i] = a * c[i] + (1 - a) * out[i - 1]
    line = pd.Series(out_fast - out_slow, index=close.index)
    sig = ema(line, signal)
    line = line.where(sig.notna())
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig}, index=close.index)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": (upper - lower) / mid}, index=close.index)


def donchian(high: pd.Series, low: pd.Series, n: int) -> pd.DataFrame:
    """Channel of the PRIOR n bars (excludes the current bar)."""
    return pd.DataFrame(
        {"dc_high": high.shift(1).rolling(n, min_periods=n).max(), "dc_low": low.shift(1).rolling(n, min_periods=n).min()},
        index=high.index,
    )


def pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Percentile (0..1) of the current value within the last n values, current included."""
    return s.rolling(n, min_periods=n).rank(pct=True)
