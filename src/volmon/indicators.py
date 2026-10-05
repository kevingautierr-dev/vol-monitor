"""Monitor indicators: IV vs RV spread, ratio, z-score, IV rank / percentile,
term-structure slope, ex-post variance risk premium and the rich/cheap signal.

Convention: vols are decimals (0.18 = 18 %). Spreads are in vol points x 100
only when displayed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import estimators as est
from .config import HORIZON, RV_FOR_SPREAD, RV_WINDOWS, Z_THRESHOLD, ZSCORE_WINDOW


def rolling_zscore(x: pd.Series, window: int = ZSCORE_WINDOW, min_periods: int | None = None) -> pd.Series:
    """z-score of today's value vs the trailing window INCLUDING today
    (no future data)."""
    mp = min_periods or window // 2
    m = x.rolling(window, min_periods=mp).mean()
    s = x.rolling(window, min_periods=mp).std(ddof=1)
    return (x - m) / s


def iv_rank(x: pd.Series, window: int = 252) -> pd.Series:
    """(IV - min) / (max - min) over the trailing year, in [0, 1]."""
    lo = x.rolling(window, min_periods=window).min()
    hi = x.rolling(window, min_periods=window).max()
    return (x - lo) / (hi - lo)


def iv_percentile(x: pd.Series, window: int = 252) -> pd.Series:
    """Share of the trailing-year days (excluding today) with IV below today."""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) > window:
        win = sliding_window_view(v, window + 1)        # past `window` days + today
        past, today = win[:, :-1], win[:, -1:]
        valid = ~np.isnan(past).any(axis=1) & ~np.isnan(today[:, 0])
        pct = (past < today).mean(axis=1)
        out[window:] = np.where(valid, pct, np.nan)
    return pd.Series(out, index=x.index, name="iv_pct_1y")


def choose_rv(ohlc: pd.DataFrame, preferred: str = RV_FOR_SPREAD, max_bad_open: float = 0.05) -> tuple[str, str]:
    """Pick the RV estimator for the spread and say why.

    Yang-Zhang needs genuine open prices. If the source copies the previous
    close into Open on more than 5 % of days, fall back to close-to-close.
    """
    q = est.data_quality(ohlc)
    if preferred.startswith(("yz", "gk", "rs", "parkinson")) and q["share_open_eq_prev_close"] > max_bad_open:
        window = preferred.split("_")[1]
        return f"cc_{window}", (f"{q['share_open_eq_prev_close']:.0%} of opens equal the prior close: "
                                f"range/overnight estimators unreliable, using close-to-close")
    return preferred, "open prices look genuine"


def build_monitor(
    ohlc: pd.DataFrame,
    iv: pd.Series,
    term: pd.DataFrame | None = None,
    rv_col: str | None = None,
    z_window: int = ZSCORE_WINDOW,
    z_thr: float = Z_THRESHOLD,
    horizon: int = HORIZON,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the daily monitor table.

    ohlc : daily bars of the underlying index
    iv   : 30-day implied vol, decimal (e.g. VIX / 100), indexed by date
    term : optional DataFrame with columns among VIX9D, VIX, VIX3M (decimals)
    """
    meta = {}
    if rv_col is None:
        rv_col, meta["rv_choice_reason"] = choose_rv(ohlc)
    meta["rv_col"] = rv_col

    rv = est.realised_vol_table(ohlc, RV_WINDOWS)
    df = rv.join(iv.rename("iv"), how="inner")
    df["close"] = ohlc["Close"].reindex(df.index)
    df["rv"] = df[rv_col]

    df["spread"] = df["iv"] - df["rv"]
    df["ratio"] = df["iv"] / df["rv"]
    df["spread_z"] = rolling_zscore(df["spread"], z_window)
    df["iv_rank_1y"] = iv_rank(df["iv"])
    df["iv_pct_1y"] = iv_percentile(df["iv"])

    # Ex-post variance risk premium: implied variance at t vs variance realised
    # over the next `horizon` days. FUTURE data -> evaluation only, never signal.
    fwd_rv2 = est.forward_realised_var(ohlc["Close"], horizon).reindex(df.index)
    df["fwd_rv"] = np.sqrt(fwd_rv2)
    df["vrp_var_expost"] = df["iv"] ** 2 - fwd_rv2
    df["vrp_vol_expost"] = df["iv"] - df["fwd_rv"]

    has_term = term is not None and {"VIX", "VIX3M"} <= set(term.columns)
    if has_term:
        t = term.reindex(df.index)
        df[[c for c in t.columns]] = t
        df["ts_slope"] = t["VIX3M"] - t["VIX"]             # > 0 contango
        df["ts_ratio"] = t["VIX"] / t["VIX3M"]             # < 1 contango
        if "VIX9D" in t:
            df["ts_short_ratio"] = t["VIX9D"] / t["VIX"]
        df["contango"] = (df["ts_slope"] > 0).where(df["ts_slope"].notna())
    meta["term_filter"] = bool(has_term)

    df["signal"] = signal(df, z_thr, use_term=has_term)
    return df, meta


def signal(df: pd.DataFrame, z_thr: float = Z_THRESHOLD, use_term: bool = True) -> pd.Series:
    """+1 'vol chère', -1 'vol bon marché', 0 neutre.

    rich  : spread z-score > +z_thr AND curve in contango (calm regime)
    cheap : spread z-score < -z_thr AND curve in backwardation (stress regime)
    Without term-structure data the curve condition is dropped (flagged in meta).
    """
    z = df["spread_z"]
    rich = z > z_thr
    cheap = z < -z_thr
    if use_term and "contango" in df:
        c = df["contango"]
        rich &= c == True        # noqa: E712  (NaN-safe)
        cheap &= c == False      # noqa: E712
    s = pd.Series(0, index=df.index, dtype="int8")
    s[rich] = 1
    s[cheap] = -1
    s[z.isna()] = 0
    return s.rename("signal")


SIGNAL_LABEL = {1: "Vol chère", -1: "Vol bon marché", 0: "Neutre"}


def latest_reading(df: pd.DataFrame, meta: dict) -> dict:
    """Plain-language summary of the last available day."""
    row = df.dropna(subset=["iv", "rv"]).iloc[-1]
    out = {
        "date": str(row.name.date()),
        "iv": round(100 * row["iv"], 2),
        "rv": round(100 * row["rv"], 2),
        "rv_estimator": meta["rv_col"],
        "spread_vol_pts": round(100 * row["spread"], 2),
        "ratio": round(row["ratio"], 2),
        "spread_z": None if pd.isna(row["spread_z"]) else round(row["spread_z"], 2),
        "iv_rank_1y": None if pd.isna(row["iv_rank_1y"]) else round(100 * row["iv_rank_1y"], 1),
        "iv_pct_1y": None if pd.isna(row["iv_pct_1y"]) else round(100 * row["iv_pct_1y"], 1),
        "signal": int(row["signal"]),
        "signal_label": SIGNAL_LABEL[int(row["signal"])],
        "term_filter": meta.get("term_filter", False),
    }
    if "ts_slope" in df and not pd.isna(row.get("ts_slope", np.nan)):
        out["ts_slope_vol_pts"] = round(100 * row["ts_slope"], 2)
        out["curve"] = "contango" if row["ts_slope"] > 0 else "backwardation"
    return out
