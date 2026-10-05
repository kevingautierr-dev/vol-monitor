"""Realised volatility estimators.

All functions take a DataFrame with columns Open, High, Low, Close (daily bars,
DatetimeIndex) and return an annualised volatility series (decimal, e.g. 0.18)
computed on a rolling window of ``window`` trading days. Value at date t only
uses bars up to and including t (no look-ahead).

References
----------
Parkinson (1980), Garman & Klass (1980), Rogers & Satchell (1991),
Yang & Zhang (2000), "Drift-independent volatility estimation based on
high, low, open, and close prices", Journal of Business 73(3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS

LN2 = np.log(2.0)


def _logs(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = (np.log(df[k].astype(float)) for k in ("Open", "High", "Low", "Close"))
    return {
        "ho": h - o,                 # high relative to open
        "lo": l - o,                 # low relative to open
        "co": c - o,                 # open-to-close (intraday) return
        "oc_prev": o - c.shift(1),   # overnight return (close t-1 -> open t)
        "cc": c - c.shift(1),        # close-to-close return
        "hl": h - l,
    }


def _annualise(var: pd.Series) -> pd.Series:
    return np.sqrt(var.clip(lower=0) * TRADING_DAYS)


def close_to_close(df: pd.DataFrame, window: int = 21, zero_mean: bool = False) -> pd.Series:
    """Classic estimator: sample std of close-to-close log returns.

    ``zero_mean=True`` gives the variance-swap convention (sum r^2 / n), used
    for the variance risk premium so that RV is comparable to VIX^2.
    """
    r = _logs(df)["cc"]
    if zero_mean:
        var = (r ** 2).rolling(window, min_periods=window).mean()
    else:
        var = r.rolling(window, min_periods=window).var(ddof=1)
    return _annualise(var).rename(f"cc_{window}")


def parkinson(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """High-low range estimator. ~5x more efficient than CC under GBM, but
    ignores overnight gaps and is biased down by discrete sampling."""
    hl = _logs(df)["hl"]
    var = (hl ** 2).rolling(window, min_periods=window).mean() / (4 * LN2)
    return _annualise(var).rename(f"parkinson_{window}")


def garman_klass(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Garman-Klass: adds open/close information to the range. Assumes zero
    drift and no overnight jump."""
    x = _logs(df)
    per_day = 0.5 * x["hl"] ** 2 - (2 * LN2 - 1) * x["co"] ** 2
    var = per_day.rolling(window, min_periods=window).mean()
    return _annualise(var).rename(f"gk_{window}")


def rogers_satchell(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Rogers-Satchell: drift-independent, still ignores overnight gap."""
    x = _logs(df)
    hc = x["ho"] - x["co"]   # ln(H/C)
    lc = x["lo"] - x["co"]   # ln(L/C)
    per_day = hc * x["ho"] + lc * x["lo"]
    var = per_day.rolling(window, min_periods=window).mean()
    return _annualise(var).rename(f"rs_{window}")


def yang_zhang(df: pd.DataFrame, window: int = 21) -> pd.Series:
    """Yang-Zhang: overnight variance + k * open-to-close variance
    + (1-k) * Rogers-Satchell. Drift-independent AND handles opening jumps;
    minimum-variance combination with k = 0.34 / (1.34 + (n+1)/(n-1))."""
    x = _logs(df)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    var_o = x["oc_prev"].rolling(n, min_periods=n).var(ddof=1)
    var_c = x["co"].rolling(n, min_periods=n).var(ddof=1)
    hc = x["ho"] - x["co"]
    lc = x["lo"] - x["co"]
    var_rs = (hc * x["ho"] + lc * x["lo"]).rolling(n, min_periods=n).mean()
    var = var_o + k * var_c + (1 - k) * var_rs
    return _annualise(var).rename(f"yz_{window}")


ESTIMATORS = {
    "cc": close_to_close,
    "parkinson": parkinson,
    "gk": garman_klass,
    "rs": rogers_satchell,
    "yz": yang_zhang,
}


def realised_vol_table(df: pd.DataFrame, windows=(10, 21, 63)) -> pd.DataFrame:
    """All estimators x all windows, columns like ``yz_21``."""
    cols = [f(df, w) for w in windows for f in ESTIMATORS.values()]
    return pd.concat(cols, axis=1)


def forward_realised_var(close: pd.Series, horizon: int = 21) -> pd.Series:
    """Annualised zero-mean realised variance over (t, t+horizon], aligned on t.

    This is the variance a 30-calendar-day variance swap struck at the close of
    t would have realised -- the right comparison for VIX_t^2. It uses FUTURE
    data by construction and must only be used for ex-post evaluation.
    """
    r = np.log(close.astype(float)).diff()
    fwd = (r ** 2).rolling(horizon, min_periods=horizon).mean().shift(-horizon)
    return (fwd * TRADING_DAYS).rename(f"fwd_rv2_{horizon}")


def data_quality(df: pd.DataFrame) -> dict:
    """Checks that matter for range-based estimators.

    Yahoo index data (e.g. ^GSPC before ~2000s, many European indices) often
    sets Open = previous Close, which silently kills the overnight term of
    Yang-Zhang. Range estimators are also broken when High == Low.
    """
    c_prev = df["Close"].shift(1)
    open_eq_prev = (np.isclose(df["Open"], c_prev, rtol=0, atol=1e-6)).mean()
    flat_bar = (df["High"] <= df["Low"]).mean()
    bad_range = ((df["High"] < df[["Open", "Close"]].max(axis=1) - 1e-9)
                 | (df["Low"] > df[["Open", "Close"]].min(axis=1) + 1e-9)).mean()
    return {
        "n_bars": int(len(df)),
        "start": str(df.index.min().date()),
        "end": str(df.index.max().date()),
        "share_open_eq_prev_close": float(open_eq_prev),
        "share_flat_bars": float(flat_bar),
        "share_inconsistent_ohlc": float(bad_range),
    }
