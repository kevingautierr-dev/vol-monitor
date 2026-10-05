"""Interactive dashboard.  streamlit run app.py

Reads data/processed (written by the daily job) and lets you change the
look-back, the RV estimator used for the spread and the z-score threshold,
recomputing the signal and its backtest on the fly.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from volmon import backtest as bt          # noqa: E402
from volmon import config as C             # noqa: E402
from volmon import indicators as ind       # noqa: E402
from volmon import report as R             # noqa: E402

st.set_page_config(page_title="Vol monitor", layout="wide")
st.title("La volatilité est-elle chère aujourd'hui ?")

summary_path = C.PROCESSED / "latest.json"
if not summary_path.exists():
    st.error("Pas de données : lancer `python scripts/daily.py` d'abord.")
    st.stop()
summary = json.loads(summary_path.read_text())
pairs = {k: v for k, v in summary["pairs"].items() if v.get("available")}

with st.sidebar:
    key = st.selectbox("Indice", list(pairs), format_func=lambda k: pairs[k]["label"])
    years = st.slider("Historique affiché (années)", 1, 20, 3)
    rv_col = st.selectbox("Estimateur RV pour le spread",
                          [f"{e}_{w}" for w in C.RV_WINDOWS for e in ("cc", "parkinson", "gk", "rs", "yz")],
                          index=[f"{e}_{w}" for w in C.RV_WINDOWS for e in ("cc", "parkinson", "gk", "rs", "yz")].index(pairs[key]["rv_col"]))
    z_win = st.select_slider("Fenêtre du z-score (jours)", [126, 252, 504], 252)
    z_thr = st.slider("Seuil de z-score", 0.5, 2.5, C.Z_THRESHOLD, 0.25)
    use_term = st.checkbox("Filtre structure par terme", value=pairs[key].get("term_filter", False),
                           disabled=not pairs[key].get("term_filter", False))
    cost = st.slider("Coût par vente (pts de vol)", 0.0, 2.0, C.COST_VOL_PTS, 0.25)
    for v in summary["pairs"].values():
        if not v.get("available"):
            st.caption(f"Non couvert : {v.get('reason')}")

df = pd.read_parquet(C.PROCESSED / f"monitor_{key}.parquet")
meta = {**pairs[key], "rv_col": rv_col, "term_filter": use_term}
df["rv"] = df[rv_col]
df["spread"] = df["iv"] - df["rv"]
df["ratio"] = df["iv"] / df["rv"]
df["spread_z"] = ind.rolling_zscore(df["spread"], z_win)
df["signal"] = ind.signal(df, z_thr, use_term=use_term)

r = ind.latest_reading(df, meta)
cols = st.columns(6)
cols[0].metric("Signal", r["signal_label"], r["date"])
cols[1].metric("IV 30j", f"{r['iv']:.1f} %")
cols[2].metric("RV", f"{r['rv']:.1f} %", rv_col, delta_color="off")
cols[3].metric("Spread", f"{r['spread_vol_pts']:+.1f} pts", f"z = {r['spread_z']:+.2f}" if r["spread_z"] is not None else "")
cols[4].metric("IV Rank / Pct 1 an", f"{r['iv_rank_1y']:.0f} / {r['iv_pct_1y']:.0f}" if r["iv_rank_1y"] is not None else "—")
cols[5].metric("Courbe", r.get("curve", "—"), f"{r['ts_slope_vol_pts']:+.1f} pts" if "ts_slope_vol_pts" in r else "")

tabs = st.tabs(["Moniteur", "Structure par terme", "Estimateurs", "Options", "Test du signal"])
with tabs[0]:
    st.plotly_chart(R.fig_iv_rv(df, meta, years), width="stretch")
    st.plotly_chart(R.fig_spread(df, z_thr, years), width="stretch")
with tabs[1]:
    f = R.fig_term(df, years)
    st.plotly_chart(f, width="stretch") if f else st.info("Pas de structure par terme pour cet indice.")
with tabs[2]:
    st.plotly_chart(R.fig_estimators(df, min(years, 5)), width="stretch")
    st.plotly_chart(R.fig_vrp(df), width="stretch")
with tabs[3]:
    if C.IV_SNAPSHOTS.exists():
        snap = pd.read_csv(C.IV_SNAPSHOTS)
        term = pd.read_csv(C.IV_TERM) if C.IV_TERM.exists() else pd.DataFrame()
        for f in R.fig_options(snap, term, df["iv"] if key == "SPX" else None):
            st.plotly_chart(f, width="stretch")
        st.dataframe(snap.tail(20), width="stretch")
    else:
        st.info("Aucun snapshot d'options encore.")
with tabs[4]:
    ex = bt.excess_vs_unconditional(df, cost=cost)
    st.markdown(R._verdict(ex))
    st.dataframe(bt.conditional_stats(df, cost=cost).round(2), width="stretch")
    st.plotly_chart(R.fig_buckets(bt.bucket_analysis(df, cost=cost)), width="stretch")
    st.plotly_chart(R.fig_equity(bt.monthly_strategies(df, cost=cost)), width="stretch")
    st.dataframe(bt.strategy_stats(df, cost=cost).round(2), width="stretch")
