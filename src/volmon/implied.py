"""Implied volatility from listed option chains.

Pipeline for one snapshot of one underlying:
1. clean quotes (positive bid, sane spread), use mid prices;
2. back out the forward of each expiry from put-call parity
   (C - P = DF * (F - K)) -> no dividend assumption needed;
3. invert Black-76 on out-of-the-money options only (puts K < F, calls K > F),
   which are the liquid ones and avoid early-exercise noise on ITM ETF options;
4. per expiry: ATM vol (interpolated in log-moneyness at K = F),
   25-delta call / put vols (interpolated in forward delta), risk reversal
   RR25 = IV(25d call) - IV(25d put) and butterfly BF25;
5. constant-maturity 30-day values: ATM interpolated linearly in total
   variance sigma^2 * T (the no-arbitrage-friendly way, same idea as VIX),
   RR/BF interpolated linearly in T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

TARGET_DAYS = 30


# --------------------------------------------------------------------------- #
# Black-76
# --------------------------------------------------------------------------- #
def black76(F, K, T, df, sigma, is_call):
    F, K, T, sigma = map(np.asarray, (F, K, T, sigma))
    sqrtT = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    call = df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    put = df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return np.where(is_call, call, put)


def forward_delta(F, K, T, sigma, is_call):
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


def implied_vol(price, F, K, T, df, is_call, lo=1e-4, hi=5.0) -> float:
    """Brent root-finding; NaN if the price violates no-arbitrage bounds."""
    intrinsic = df * max(F - K, 0.0) if is_call else df * max(K - F, 0.0)
    upper = df * F if is_call else df * K
    if not (intrinsic < price < upper):
        return np.nan
    f = lambda s: float(black76(F, K, T, df, s, is_call)) - price
    try:
        return brentq(f, lo, hi, xtol=1e-8)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------------- #
# Chain processing
# --------------------------------------------------------------------------- #
def clean_quotes(chain: pd.DataFrame, max_rel_spread: float = 0.5) -> pd.DataFrame:
    """Expects columns strike, bid, ask, type ('C'/'P'). Adds ``mid``."""
    c = chain.copy()
    c = c[(c["bid"] > 0) & (c["ask"] > c["bid"])]
    c["mid"] = 0.5 * (c["bid"] + c["ask"])
    c = c[(c["ask"] - c["bid"]) / c["mid"] <= max_rel_spread]
    return c


def implied_forward(chain: pd.DataFrame, df: float, spot: float, n: int = 5) -> float:
    """Median of K + (C - P)/DF over the n strikes where |C - P| is smallest."""
    piv = chain.pivot_table(index="strike", columns="type", values="mid")
    if not {"C", "P"} <= set(piv.columns):
        return np.nan
    piv = piv.dropna()
    if piv.empty:
        return np.nan
    piv = piv.assign(diff=(piv["C"] - piv["P"]).abs()).nsmallest(n, "diff")
    fwd = piv.index.values + (piv["C"] - piv["P"]).values / df
    f = float(np.median(fwd))
    # guard against garbage quotes
    return f if 0.8 * spot < f < 1.2 * spot else np.nan


def expiry_smile(chain: pd.DataFrame, T: float, r: float, spot: float) -> pd.DataFrame:
    """OTM implied vols for one expiry, with log-moneyness and forward delta."""
    df = np.exp(-r * T)
    q = clean_quotes(chain)
    F = implied_forward(q, df, spot)
    if np.isnan(F):
        return pd.DataFrame()
    otm = q[((q["type"] == "P") & (q["strike"] < F)) | ((q["type"] == "C") & (q["strike"] >= F))].copy()
    otm["iv"] = [
        implied_vol(p, F, k, T, df, t == "C")
        for p, k, t in zip(otm["mid"], otm["strike"], otm["type"])
    ]
    otm = otm.dropna(subset=["iv"])
    otm = otm[(otm["iv"] > 0.01) & (otm["iv"] < 3.0)]
    otm["F"] = F
    otm["T"] = T
    otm["k"] = np.log(otm["strike"] / F)
    otm["delta"] = forward_delta(F, otm["strike"].values, T, otm["iv"].values, otm["type"].values == "C")
    return otm.sort_values("strike")


def smile_metrics(smile: pd.DataFrame) -> dict:
    """ATM / 25-delta metrics for one expiry."""
    if len(smile) < 5:
        return {}
    s = smile.sort_values("k")
    w = s["iv"] ** 2 * s["T"]                          # total variance
    atm_w = np.interp(0.0, s["k"], w)
    T = float(s["T"].iloc[0])
    out = {"T": T, "F": float(s["F"].iloc[0]), "atm_iv": float(np.sqrt(atm_w / T)), "n_quotes": int(len(s))}

    calls = s[s["type"] == "C"].sort_values("delta")    # delta ascending
    puts = s[s["type"] == "P"].sort_values("delta")
    if len(calls) >= 2 and calls["delta"].min() <= 0.25 <= calls["delta"].max():
        out["iv_25c"] = float(np.interp(0.25, calls["delta"], calls["iv"]))
    if len(puts) >= 2 and puts["delta"].min() <= -0.25 <= puts["delta"].max():
        out["iv_25p"] = float(np.interp(-0.25, puts["delta"], puts["iv"]))
    if "iv_25c" in out and "iv_25p" in out:
        out["rr25"] = out["iv_25c"] - out["iv_25p"]
        out["bf25"] = 0.5 * (out["iv_25c"] + out["iv_25p"]) - out["atm_iv"]
    return out


def constant_maturity(term: pd.DataFrame, days: int = TARGET_DAYS) -> dict:
    """Interpolate per-expiry metrics to a fixed horizon.

    ``term``: one row per expiry with columns T, atm_iv, rr25, bf25.
    ATM: linear in total variance; RR/BF: linear in T; flat extrapolation.
    """
    t = days / 365.0
    term = term.dropna(subset=["atm_iv"]).sort_values("T")
    if term.empty:
        return {}
    w = term["atm_iv"] ** 2 * term["T"]
    if t <= term["T"].iloc[0]:
        atm = term["atm_iv"].iloc[0]
    elif t >= term["T"].iloc[-1]:
        atm = term["atm_iv"].iloc[-1]
    else:
        atm = np.sqrt(np.interp(t, term["T"], w) / t)
    out = {f"atm_iv_{days}d": float(atm)}
    for col in ("rr25", "bf25", "iv_25c", "iv_25p"):
        if col in term and term[col].notna().sum() >= 1:
            sub = term.dropna(subset=[col])
            out[f"{col}_{days}d"] = float(np.interp(t, sub["T"], sub[col]))
    return out


def term_structure(chains: dict[float, pd.DataFrame], r: float, spot: float) -> pd.DataFrame:
    """``chains``: {T (years): chain DataFrame}. Returns one row per expiry."""
    rows = []
    for T, ch in sorted(chains.items()):
        sm = expiry_smile(ch, T, r, spot)
        m = smile_metrics(sm)
        if m:
            rows.append(m)
    return pd.DataFrame(rows)
