"""Does a rich IV-RV spread precede a winning vol sale?

Trade proxy: short a 30-day variance swap on the index at the close of day t,
struck at the implied vol (VIX is, by construction, the fair strike of a
30-day variance swap, up to discrete-strike and jump corrections), settled on
the zero-mean realised variance of the next ``horizon`` trading days.

P&L is expressed per unit of VEGA notional, in vol points:

    pnl_t = (K_t^2 - RV_{t,t+h}^2) / (2 K_t) * 100 - cost

so a 1-vol-point realisation below the strike is worth ~1 point, but a vol
spike costs much more (the convexity that makes short variance a
negatively-skewed, crash-exposed trade).

Statistics are computed on overlapping daily observations with Newey-West
standard errors (lag = horizon - 1) AND on a non-overlapping monthly sample,
because 21-day overlapping returns otherwise inflate t-stats ~sqrt(21)-fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_VOL_PTS, HORIZON


def short_variance_pnl(df: pd.DataFrame, cost: float = COST_VOL_PTS) -> pd.Series:
    K = df["iv"]
    rv2 = df["fwd_rv"] ** 2
    return ((K ** 2 - rv2) / (2 * K) * 100 - cost).rename("pnl")


def newey_west_tstat(x: pd.Series, lag: int) -> float:
    x = x.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    s = e @ e / n
    for j in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - j / (lag + 1)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else np.nan


def _stats(p: pd.Series, lag: int) -> dict:
    p = p.dropna()
    if p.empty:
        return {"n": 0}
    q05 = p.quantile(0.05)
    return {
        "n": int(len(p)),
        "mean": float(p.mean()),
        "median": float(p.median()),
        "hit_rate": float((p > 0).mean()),
        "std": float(p.std()),
        "t_nw": newey_west_tstat(p, lag),
        "worst": float(p.min()),
        "cvar5": float(p[p <= q05].mean()),
        "skew": float(p.skew()),
    }


def conditional_stats(df: pd.DataFrame, horizon: int = HORIZON, cost: float = COST_VOL_PTS) -> pd.DataFrame:
    """Forward short-variance P&L by signal state, overlapping (NW) and monthly."""
    d = df.assign(pnl=short_variance_pnl(df, cost)).dropna(subset=["pnl", "spread_z"])
    out = {}
    groups = {"Toujours vendre": d, "Vol chère (+1)": d[d.signal == 1],
              "Neutre (0)": d[d.signal == 0], "Vol bon marché (-1)": d[d.signal == -1]}
    for name, g in groups.items():
        out[name] = _stats(g["pnl"], horizon - 1)
        mon = g.iloc[::horizon] if name == "Toujours vendre" else g[g.index.isin(d.index[::horizon])]
        m = mon["pnl"].dropna()
        out[name]["n_monthly"] = int(len(m))
        out[name]["mean_monthly"] = float(m.mean()) if len(m) else np.nan
        out[name]["t_monthly"] = (float(m.mean() / (m.std(ddof=1) / np.sqrt(len(m))))
                                  if len(m) > 2 and m.std() > 0 else np.nan)
    return pd.DataFrame(out).T


def excess_vs_unconditional(df: pd.DataFrame, horizon: int = HORIZON, cost: float = COST_VOL_PTS) -> dict:
    """The honest question: does conditioning on the signal beat ALWAYS selling?
    Regress pnl on a rich-dummy and a cheap-dummy; NW t-stats on the dummies."""
    d = df.assign(pnl=short_variance_pnl(df, cost)).dropna(subset=["pnl", "spread_z"])
    y = d["pnl"].to_numpy()
    names = ["const_neutral", "rich_excess", "cheap_excess"]
    cols = [np.ones(len(d)), (d.signal == 1).to_numpy(float), (d.signal == -1).to_numpy(float)]
    keep = [i for i, c in enumerate(cols) if i == 0 or c.sum() >= 5]   # drop empty states
    X = np.column_stack([cols[i] for i in keep])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    L = horizon - 1
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for j in range(1, L + 1):
        G = Xu[j:].T @ Xu[:-j]
        S += (1 - j / (L + 1)) * (G + G.T)
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    tnames = ["t_const", "t_rich", "t_cheap"]
    out = {n: np.nan for n in names + tnames}
    for pos, i in enumerate(keep):
        out[names[i]] = beta[pos]
        out[tnames[i]] = beta[pos] / se[pos]
    out["n"] = len(d)
    return out


def bucket_analysis(df: pd.DataFrame, col: str = "spread_z", n_buckets: int = 5,
                    horizon: int = HORIZON, cost: float = COST_VOL_PTS) -> pd.DataFrame:
    """Mean forward P&L and ex-post VRP by quintile of a predictor.
    A useful signal shows a monotonic pattern; buckets are formed on the full
    sample (descriptive, not tradeable -- the threshold rule is the tradeable one)."""
    d = df.assign(pnl=short_variance_pnl(df, cost)).dropna(subset=["pnl", col])
    d["bucket"] = pd.qcut(d[col], n_buckets, labels=[f"Q{i+1}" for i in range(n_buckets)])
    g = d.groupby("bucket", observed=True)
    res = g.agg(lo=(col, "min"), hi=(col, "max"), n=("pnl", "size"),
                mean_pnl=("pnl", "mean"), hit_rate=("pnl", lambda x: (x > 0).mean()),
                worst=("pnl", "min"), vrp_vol=("vrp_vol_expost", "mean"))
    res["t_nw"] = g["pnl"].apply(lambda x: newey_west_tstat(x, horizon - 1))
    res["vrp_vol"] *= 100
    return res


def robustness_grid(df: pd.DataFrame, thresholds=(0.5, 1.0, 1.5, 2.0), z_windows=(126, 252, 504),
                    use_term: bool = True, horizon: int = HORIZON) -> pd.DataFrame:
    """Excess P&L of the signal states over neutral days, for several z-score
    thresholds and look-back windows -- a result that only holds at one
    parameter is noise. (A uniform cost shifts every state equally and leaves
    the excess unchanged, so it is not part of the grid.)"""
    from .indicators import rolling_zscore, signal
    rows = []
    for w in z_windows:
        d = df.copy()
        d["spread_z"] = rolling_zscore(d["spread"], w)
        for thr in thresholds:
            d["signal"] = signal(d, thr, use_term=use_term and "contango" in d)
            r = excess_vs_unconditional(d, horizon)
            rows.append({"fenêtre_z": w, "z_seuil": thr, "n_riche": int((d.signal == 1).sum()),
                         "n_bon_marché": int((d.signal == -1).sum()), "pnl_neutre": r["const_neutral"],
                         "excès_riche": r["rich_excess"], "t_riche": r["t_rich"],
                         "excès_bon_marché": r["cheap_excess"], "t_bon_marché": r["t_cheap"]})
    return pd.DataFrame(rows)


def subperiod_stats(df: pd.DataFrame, splits: list[str], horizon: int = HORIZON) -> pd.DataFrame:
    """Same regression on sub-periods: stability check in lieu of a true out-of-sample."""
    bounds = [df.index.min()] + [pd.Timestamp(s) for s in splits] + [df.index.max() + pd.Timedelta(days=1)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = df[(df.index >= a) & (df.index < b)]
        if sub["signal"].abs().sum() < 20:
            continue
        r = excess_vs_unconditional(sub, horizon)
        rows.append({"période": f"{a.year}–{(b - pd.Timedelta(days=1)).year}", "n": r["n"],
                     "pnl_neutre": r["const_neutral"], "excès_riche": r["rich_excess"], "t_riche": r["t_rich"],
                     "excès_bon_marché": r["cheap_excess"], "t_bon_marché": r["t_cheap"]})
    return pd.DataFrame(rows)


def monthly_strategies(df: pd.DataFrame, horizon: int = HORIZON, cost: float = COST_VOL_PTS) -> pd.DataFrame:
    """Non-overlapping roll every `horizon` days: cumulative P&L (vol points per
    unit vega) of (a) always short, (b) short only when 'rich', (c) short
    unless 'cheap'. Uses only information available at each roll date."""
    d = df.assign(pnl=short_variance_pnl(df, cost)).dropna(subset=["pnl", "spread_z"]).iloc[::horizon]
    strat = pd.DataFrame(index=d.index)
    strat["Toujours vendre"] = d["pnl"]
    strat["Vendre si vol chère"] = d["pnl"].where(d.signal == 1, 0.0)
    strat["Vendre sauf si bon marché"] = d["pnl"].where(d.signal != -1, 0.0)
    return strat.cumsum()


def strategy_stats(df: pd.DataFrame, horizon: int = HORIZON, cost: float = COST_VOL_PTS) -> pd.DataFrame:
    """Per-trade view of the monthly strategies: the 'rich only' strategy trades
    far less often, so total P&L alone is an unfair comparison."""
    eq = monthly_strategies(df, horizon, cost)
    pnl = eq.diff().fillna(eq.iloc[0])
    d = df.dropna(subset=["fwd_rv", "spread_z"]).iloc[::horizon]
    active = {"Toujours vendre": pd.Series(True, index=d.index),
              "Vendre si vol chère": d.signal == 1,
              "Vendre sauf si bon marché": d.signal != -1}
    rows = {}
    per_year = 252 / horizon
    for c in pnl.columns:
        tr = pnl[c][active[c].reindex(pnl.index, fill_value=False)]
        dd = (eq[c] - eq[c].cummax()).min()
        rows[c] = {"trades": len(tr), "pnl_total": eq[c].iloc[-1], "pnl_par_trade": tr.mean(),
                   "hit_rate": (tr > 0).mean(), "pire_trade": tr.min(),
                   "sharpe_annualisé": (pnl[c].mean() / pnl[c].std() * np.sqrt(per_year)) if pnl[c].std() > 0 else np.nan, "max_drawdown": dd}
    return pd.DataFrame(rows).T
