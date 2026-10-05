"""Reproducible study on data that does not need Yahoo:
- S&P 500 daily OHLC 1999-2018 bundled with the `arch` package (originally Yahoo ^GSPC)
- VIX daily close 1990-today from CBOE, mirrored at github.com/datasets/finance-vix

No VIX3M on this source, so the term-structure filter is OFF here: the rule
tested is the z-score leg alone. The full rule (z-score + contango) is tested
by the daily job on Yahoo data (VIX3M from 2006).

    python scripts/offline_study.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volmon import backtest as bt          # noqa: E402
from volmon import estimators as est       # noqa: E402
from volmon import indicators as ind       # noqa: E402
from volmon.report import build_page       # noqa: E402

VIX_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"
VIX_CSV = ROOT / "data" / "external" / "vix_cboe.csv"
OUT = ROOT / "results"


def load():
    from arch.data import sp500
    spx = sp500.load()[["Open", "High", "Low", "Close"]]
    spx.index = pd.to_datetime(spx.index)
    if not VIX_CSV.exists():
        VIX_CSV.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(VIX_URL, VIX_CSV)
    vix = pd.read_csv(VIX_CSV, parse_dates=["DATE"], index_col="DATE")["CLOSE"] / 100
    return spx, vix


def main():
    spx, vix = load()
    OUT.mkdir(exist_ok=True)

    # 1. data quality per year: where can range estimators be trusted?
    dq = pd.DataFrame({y: est.data_quality(g) for y, g in spx.groupby(spx.index.year)}).T
    dq = dq[["n_bars", "share_open_eq_prev_close"]]
    dq.to_csv(OUT / "data_quality_by_year.csv")

    # 2. monitor (auto-selects close-to-close because opens are unreliable)
    df, meta = ind.build_monitor(spx, vix)
    meta["label"] = "S&P 500 / VIX — étude 1999-2018"

    # 3. estimator comparison on the period with genuine opens only
    good = spx[spx.index >= "2014-01-01"]
    rv = est.realised_vol_table(good, (21,)).dropna()
    fwd = est.forward_realised_var(good["Close"], 21).pow(0.5).reindex(rv.index)
    est_cmp = pd.DataFrame({
        "moyenne_%": 100 * rv.mean(),
        "écart-type_%": 100 * rv.std(),
        "corr_avec_RV_future": rv.corrwith(fwd),
        "RMSE_vs_RV_future_pts": 100 * ((rv.sub(fwd, axis=0)) ** 2).mean() ** 0.5,
    }).round(3)
    est_cmp.to_csv(OUT / "estimators_2014_2018.csv")

    # 4. signal test
    cs = bt.conditional_stats(df)
    ex = bt.excess_vs_unconditional(df)
    bk = bt.bucket_analysis(df)
    rob = bt.robustness_grid(df, use_term=False)
    sub = bt.subperiod_stats(df, ["2008-01-01", "2013-01-01"])
    cs.to_csv(OUT / "conditional_stats.csv"); bk.to_csv(OUT / "buckets.csv")
    rob.to_csv(OUT / "robustness.csv", index=False); sub.to_csv(OUT / "subperiods.csv", index=False)

    v = df.dropna(subset=["vrp_vol_expost"])
    summary = {
        "sample": [str(df.index.min().date()), str(df.index.max().date())],
        "rv_estimator": meta["rv_col"], "rv_reason": meta["rv_choice_reason"],
        "mean_iv": 100 * df["iv"].mean(), "mean_rv": 100 * df["rv"].mean(),
        "vrp_vol_mean_pts": 100 * v["vrp_vol_expost"].mean(),
        "vrp_positive_share": float((v["vrp_vol_expost"] > 0).mean()),
        "vrp_var_mean": float(v["vrp_var_expost"].mean()),
        "excess_regression": {k: float(x) for k, x in ex.items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    banner = ("Étude reproductible hors Yahoo : S&amp;P 500 OHLC 1999-2018 (paquet <code>arch</code>) + VIX CBOE. "
              "Pas de VIX3M sur cette source : la règle testée ici est le z-score seul, sans filtre de courbe. "
              "Le moniteur quotidien (index.html) applique la règle complète sur données Yahoo.")
    build_page({"SPX": (df, meta)}, ROOT / "docs" / "study_spx_1999_2018.html",
               "Étude du signal — S&P 500 1999-2018",
               "Prime implicite − réalisée et vente de variance : résultats, y compris négatifs.",
               banner=banner, splits=["2008-01-01", "2013-01-01"])

    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print(dq.round(3).to_string())
    print(est_cmp.to_string())
    print(cs.round(2).to_string())
    print(bk.round(2).to_string())
    print(rob.round(2).to_string())
    print(sub.round(2).to_string())


if __name__ == "__main__":
    main()
