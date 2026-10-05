"""Static HTML dashboard (Plotly) -- served by GitHub Pages from docs/.

One self-contained page: today's reading, IV vs RV, spread z-score and
signal, VIX term structure, estimator comparison, ex-post VRP, option-chain
metrics (SPY/QQQ snapshots) and the signal test with its negative results.
"""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import backtest as bt
from . import config as C
from .indicators import SIGNAL_LABEL, latest_reading

NAVY, GOLD, CREAM = "#16293E", "#A9802E", "#FBFAF7"
RED, GREEN, GREY, BLUE = "#B5473A", "#3C7A5A", "#8A8F98", "#4A78A8"
PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"

LAYOUT = dict(
    template="simple_white", font=dict(family="Inter, Helvetica, Arial", size=12, color=NAVY),
    margin=dict(l=55, r=25, t=85, b=40), height=380, hovermode="x unified",
    title=dict(x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)),
    legend=dict(orientation="h", y=-0.12, yanchor="top", x=0), paper_bgcolor="white", plot_bgcolor="white",
)


def _fig_html(fig: go.Figure) -> str:
    own = fig.layout.to_plotly_json()       # keep per-figure overrides (height, hovermode...)
    upd = {k: v for k, v in LAYOUT.items() if k not in own or k in ("template", "title", "margin", "legend")}
    fig.update_layout(**upd)
    if fig.layout.showlegend is not False:
        fig.update_layout(margin=dict(b=80))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False, "responsive": True})


def _last_years(df: pd.DataFrame, years: float) -> pd.DataFrame:
    return df[df.index >= df.index.max() - pd.DateOffset(days=int(365 * years))]


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def fig_iv_rv(df: pd.DataFrame, meta: dict, years: float = 3) -> go.Figure:
    d = _last_years(df, years)
    f = go.Figure()
    f.add_scatter(x=d.index, y=100 * d["iv"], name="Vol implicite 30j", line=dict(color=NAVY, width=2))
    f.add_scatter(x=d.index, y=100 * d["rv"], name=f"Vol réalisée ({meta['rv_col']})", line=dict(color=GOLD, width=2))
    f.add_scatter(x=d.index, y=100 * d["cc_63"], name="Vol réalisée 63j", line=dict(color=GREY, width=1, dash="dot"))
    f.update_layout(title="Implicite vs réalisée (%)", yaxis_title="vol %")
    return f


def fig_spread(df: pd.DataFrame, z_thr: float, years: float = 3) -> go.Figure:
    d = _last_years(df, years)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06)
    f.add_scatter(x=d.index, y=100 * d["spread"], name="Spread IV − RV (pts)", line=dict(color=NAVY), row=1, col=1)
    f.add_hline(y=0, line=dict(color=GREY, width=1), row=1, col=1)
    f.add_scatter(x=d.index, y=d["spread_z"], name="z-score 1 an", line=dict(color=GOLD), row=2, col=1)
    for y in (z_thr, -z_thr):
        f.add_hline(y=y, line=dict(color=GREY, dash="dash", width=1), row=2, col=1)
    rich, cheap = d[d.signal == 1], d[d.signal == -1]
    f.add_scatter(x=rich.index, y=rich["spread_z"], mode="markers", name="Vol chère",
                  marker=dict(color=RED, size=5), row=2, col=1)
    f.add_scatter(x=cheap.index, y=cheap["spread_z"], mode="markers", name="Vol bon marché",
                  marker=dict(color=GREEN, size=5), row=2, col=1)
    f.update_layout(title="Prime implicite − réalisée et signal", height=460)
    return f


def fig_term(df: pd.DataFrame, years: float = 3) -> go.Figure | None:
    if "ts_ratio" not in df:
        return None
    d = _last_years(df, years)
    last = df.dropna(subset=["VIX", "VIX3M"]).iloc[-1]
    f = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                      subplot_titles=("VIX / VIX3M (< 1 = contango)", f"Courbe au {last.name.date()}"))
    f.add_scatter(x=d.index, y=d["ts_ratio"], name="VIX/VIX3M", line=dict(color=NAVY), row=1, col=1)
    if "ts_short_ratio" in d:
        f.add_scatter(x=d.index, y=d["ts_short_ratio"], name="VIX9D/VIX", line=dict(color=GOLD, width=1), row=1, col=1)
    f.add_hline(y=1, line=dict(color=RED, dash="dash", width=1), row=1, col=1)
    pts = [(9, "VIX9D"), (30, "VIX"), (93, "VIX3M"), (182, "VIX6M")]
    pts = [(x, k) for x, k in pts if k in last and pd.notna(last[k])]
    f.add_scatter(x=[p[0] for p in pts], y=[100 * last[p[1]] for p in pts], text=[p[1] for p in pts],
                  mode="lines+markers+text", textposition="top center", name="Courbe",
                  line=dict(color=GOLD, width=2), marker=dict(size=8), row=1, col=2)
    f.update_xaxes(title_text="jours", row=1, col=2)
    f.update_layout(title="Structure par terme", showlegend=True)
    return f


def fig_estimators(df: pd.DataFrame, years: float = 1) -> go.Figure:
    d = _last_years(df, years)
    f = go.Figure()
    colors = {"cc_21": NAVY, "parkinson_21": BLUE, "gk_21": GREEN, "rs_21": GREY, "yz_21": GOLD}
    names = {"cc_21": "Close-to-close", "parkinson_21": "Parkinson", "gk_21": "Garman-Klass",
             "rs_21": "Rogers-Satchell", "yz_21": "Yang-Zhang"}
    for c, col in colors.items():
        f.add_scatter(x=d.index, y=100 * d[c], name=names[c], line=dict(color=col, width=1.6))
    f.update_layout(title="Estimateurs de vol réalisée, fenêtre 21j (%)")
    return f


def fig_vrp(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["vrp_vol_expost"])
    f = make_subplots(rows=1, cols=2, column_widths=[0.68, 0.32],
                      subplot_titles=("IV(t) − RV réalisée sur t→t+21 (pts)", "Distribution"))
    f.add_scatter(x=d.index, y=100 * d["vrp_vol_expost"], name="VRP ex post",
                  line=dict(color=NAVY, width=1), row=1, col=1)
    f.add_hline(y=0, line=dict(color=RED, width=1), row=1, col=1)
    f.add_histogram(x=100 * d["vrp_vol_expost"], nbinsx=80, marker_color=GOLD, name="fréquence", row=1, col=2)
    f.update_layout(title=f"Prime de risque de variance ex post — positive {100 * (d['vrp_vol_expost'] > 0).mean():.0f} % du temps",
                    showlegend=False, hovermode="closest")
    return f


def fig_buckets(b: pd.DataFrame) -> go.Figure:
    f = make_subplots(rows=1, cols=2, subplot_titles=("P&L moyen à 21j (pts de vol, par unité de vega)",
                                                      "Pire perte (pts)"))
    labels = [f"{i}<br>[{r.lo:.1f}; {r.hi:.1f}]" for i, r in b.iterrows()]
    f.add_bar(x=labels, y=b["mean_pnl"], marker_color=NAVY, text=[f"{v:.2f}<br>t={t:.1f}" for v, t in zip(b.mean_pnl, b.t_nw)],
              textposition="outside", row=1, col=1)
    f.add_bar(x=labels, y=b["worst"], marker_color=RED, text=[f"{v:.0f}" for v in b.worst], textposition="outside", row=1, col=2)
    f.update_layout(title="Vente de variance selon le quintile de z-score du spread", showlegend=False, hovermode="closest")
    return f


def fig_equity(eq: pd.DataFrame) -> go.Figure:
    f = go.Figure()
    for c, col in zip(eq.columns, (NAVY, GOLD, GREY)):
        f.add_scatter(x=eq.index, y=eq[c], name=c, line=dict(color=col, width=2))
    f.update_layout(title="P&L cumulé, roll mensuel non chevauchant (pts de vol par unité de vega)")
    return f


def fig_options(snap: pd.DataFrame, term: pd.DataFrame, vix: pd.Series | None) -> list[go.Figure]:
    figs = []
    if snap.empty:
        return figs
    snap = snap.copy(); snap["date"] = pd.to_datetime(snap["date"])
    f = make_subplots(rows=1, cols=2, subplot_titles=("ATM 30j (%)", "Risk reversal 25Δ 30j (pts)"))
    for (t, g), col in zip(snap.groupby("ticker"), (NAVY, GOLD, GREEN)):
        f.add_scatter(x=g["date"], y=100 * g["atm_iv_30d"], name=f"{t} ATM 30j", mode="lines+markers",
                      line=dict(color=col), row=1, col=1)
        if "rr25_30d" in g:
            f.add_scatter(x=g["date"], y=100 * g["rr25_30d"], name=f"{t} RR25", mode="lines+markers",
                          line=dict(color=col, dash="dot"), row=1, col=2)
    if vix is not None:
        v = vix[vix.index >= snap["date"].min()]
        f.add_scatter(x=v.index, y=100 * v, name="VIX", line=dict(color=GREY, dash="dash"), row=1, col=1)
    f.update_layout(title="Vol implicite reconstruite depuis les chaînes d'options")
    figs.append(f)
    if not term.empty:
        last_day = term["date"].max()
        t = term[term["date"] == last_day]
        f2 = go.Figure()
        for (tk, g), col in zip(t.groupby("ticker"), (NAVY, GOLD, GREEN)):
            f2.add_scatter(x=g["dte"], y=100 * g["atm_iv"], name=f"{tk} ATM", mode="lines+markers", line=dict(color=col))
            if "rr25" in g:
                f2.add_scatter(x=g["dte"], y=100 * g["rr25"], name=f"{tk} RR25", mode="lines+markers",
                               line=dict(color=col, dash="dot"), yaxis="y2")
        f2.update_layout(title=f"Structure par terme des options au {last_day}", xaxis_title="jours",
                         yaxis_title="ATM %", yaxis2=dict(title="RR25 pts", overlaying="y", side="right"))
        figs.append(f2)
    return figs


# --------------------------------------------------------------------------- #
# Tables / cards
# --------------------------------------------------------------------------- #
def _table(df: pd.DataFrame, fmt: dict | None = None, index=True) -> str:
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt:
            d[c] = d[c].map(lambda v, f=fmt[c]: "" if pd.isna(v) else f.format(v))
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    return d.to_html(index=index, classes="tbl", border=0, escape=True)


def _card(label, value, sub=""):
    return f'<div class="card"><div class="lbl">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>'


def reading_cards(r: dict) -> str:
    badge = {1: "rich", -1: "cheap", 0: "neutral"}[r["signal"]]
    cards = [
        f'<div class="card sig {badge}"><div class="lbl">Signal au {r["date"]}</div>'
        f'<div class="val">{html.escape(r["signal_label"])}</div>'
        f'<div class="sub">{"z-score + structure par terme" if r["term_filter"] else "z-score seul (pas de courbe)"}</div></div>',
        _card("Vol implicite 30j", f"{r['iv']:.1f} %"),
        _card("Vol réalisée", f"{r['rv']:.1f} %", r["rv_estimator"]),
        _card("Spread IV − RV", f"{r['spread_vol_pts']:+.1f} pts", f"ratio {r['ratio']:.2f}"),
        _card("z-score du spread", "—" if r["spread_z"] is None else f"{r['spread_z']:+.2f}", "fenêtre 1 an"),
        _card("IV Rank / Percentile", f"{r['iv_rank_1y']:.0f} / {r['iv_pct_1y']:.0f}" if r["iv_rank_1y"] is not None else "—", "1 an"),
    ]
    if "curve" in r:
        cards.append(_card("Courbe VIX", r["curve"], f"VIX3M − VIX = {r['ts_slope_vol_pts']:+.1f} pts"))
    return '<div class="cards">' + "".join(cards) + "</div>"


CSS = f"""
*{{box-sizing:border-box}} body{{margin:0;background:{CREAM};color:{NAVY};font-family:Inter,Helvetica,Arial,sans-serif;line-height:1.5}}
header{{background:{NAVY};color:white;padding:28px 16px}} header .in, main{{max-width:1180px;margin:0 auto}}
header h1{{margin:0;font-size:24px;font-weight:600}} header p{{margin:6px 0 0;color:#cfd6df;font-size:14px}}
.kicker{{color:{GOLD};letter-spacing:.18em;font-size:11px;text-transform:uppercase}}
main{{padding:16px}} section{{background:white;border:1px solid #e8e4da;border-radius:6px;padding:18px;margin:16px 0}}
h2{{font-size:18px;margin:0 0 4px}} h3{{font-size:15px;margin:18px 0 6px}} .note{{color:#5b6573;font-size:13px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}}
.card{{background:{CREAM};border:1px solid #e8e4da;border-radius:6px;padding:10px 12px}}
.card .lbl{{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6b7280}}
.card .val{{font-size:22px;font-weight:600;margin-top:2px}} .card .sub{{font-size:12px;color:#6b7280}}
.sig.rich{{border-left:5px solid {RED}}} .sig.cheap{{border-left:5px solid {GREEN}}} .sig.neutral{{border-left:5px solid {GREY}}}
.tbl{{border-collapse:collapse;font-size:12.5px;width:100%;overflow-x:auto;display:block}}
.tbl th,.tbl td{{padding:5px 8px;border-bottom:1px solid #eee;text-align:right;white-space:nowrap}}
.tbl th{{color:#6b7280;font-weight:600}} .tbl tbody tr th{{text-align:left;color:{NAVY}}}
.verdict{{border-left:4px solid {GOLD};background:{CREAM};padding:10px 14px;margin:12px 0;font-size:14px}}
.warn{{border-left:4px solid {RED};background:#fbf1ef;padding:10px 14px;margin:12px 0;font-size:13.5px}}
footer{{max-width:1180px;margin:0 auto;padding:16px;font-size:12px;color:#6b7280}}
"""


def _verdict(ex: dict) -> str:
    def word(t):
        return "significatif (|t| > 2)" if abs(t) > 2 else ("faible (1 < |t| < 2)" if abs(t) > 1 else "non significatif")

    def part(label, b, t):
        if pd.isna(b):
            return f"aucun jour « {label} » sur l'échantillon"
        return f"les jours « {label} » : {b:+.2f} pt (t = {t:.1f}, {word(t)})"
    return (f"Vendre la variance les jours neutres rapporte {ex['const_neutral']:.2f} pt (t = {ex['t_const']:.1f}). "
            f"Écart vs neutre — {part('vol chère', ex['rich_excess'], ex['t_rich'])} ; "
            f"{part('vol bon marché', ex['cheap_excess'], ex['t_cheap'])}.")


def signal_study_html(df: pd.DataFrame, meta: dict, splits: list[str]) -> str:
    cs = bt.conditional_stats(df)
    ex = bt.excess_vs_unconditional(df)
    b = bt.bucket_analysis(df)
    eq = bt.monthly_strategies(df)
    rob = bt.robustness_grid(df, use_term=meta.get("term_filter", False))
    sub = bt.subperiod_stats(df, splits)
    start = df.dropna(subset=["spread_z", "fwd_rv"]).index.min().date()
    end = df.dropna(subset=["fwd_rv"]).index.max().date()
    cs_cols = {"n": "{:.0f}", "mean": "{:.2f}", "median": "{:.2f}", "hit_rate": "{:.0%}", "std": "{:.2f}",
               "t_nw": "{:.2f}", "worst": "{:.1f}", "cvar5": "{:.1f}", "skew": "{:.2f}",
               "n_monthly": "{:.0f}", "mean_monthly": "{:.2f}", "t_monthly": "{:.2f}"}
    cs = cs.rename(columns={"t_nw": "t_nw", "hit_rate": "hit_rate"})
    return f"""
<p class="note">Échantillon {start} → {end}. Trade : vente d'un variance swap 30j au strike IV(t), réglé sur la variance réalisée
des 21 séances suivantes ; P&amp;L par unité de vega notionnel, en points de vol, coût {C.COST_VOL_PTS} pt déduit.
Filtre de courbe : {"oui" if meta.get("term_filter") else "<b>non</b> (pas de données VIX3M sur la période)"}.</p>
<div class="verdict">{_verdict(ex)}</div>
<h3>P&amp;L à 21 jours par état du signal</h3>{_table(cs, cs_cols)}
<p class="note">t_nw : t-stat Newey-West (lag 20) sur observations quotidiennes chevauchantes ; *_monthly : un trade toutes les 21 séances,
sans chevauchement. cvar5 : perte moyenne des 5 % pires cas.</p>
{_fig_html(fig_buckets(b))}
{_fig_html(fig_equity(eq))}
{_table(bt.strategy_stats(df), {"trades": "{:.0f}", "hit_rate": "{:.0%}", "pnl_total": "{:.0f}", "max_drawdown": "{:.1f}", "pire_trade": "{:.1f}"})}
<p class="note">Sharpe calculé sur toutes les dates de roll (0 quand la stratégie est à plat). Le VIX n'est pas négociable :
un vrai variance swap se traite au-dessus du VIX (convexité, sauts) et avec une fourchette — d'où le coût déduit, qui reste une approximation.</p>
<h3>Robustesse : seuil et fenêtre du z-score</h3>{_table(rob, {"n_riche": "{:.0f}", "n_bon_marché": "{:.0f}", "fenêtre_z": "{:.0f}"}, index=False)}
<h3>Stabilité par sous-période</h3>{_table(sub, {"n": "{:.0f}"}, index=False) if not sub.empty else "<p class='note'>Pas assez de signaux.</p>"}
"""


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def build_page(monitors: dict[str, tuple[pd.DataFrame, dict]], out: Path, title: str, subtitle: str,
               snap: pd.DataFrame | None = None, term_snap: pd.DataFrame | None = None,
               banner: str = "", splits: list[str] | None = None, study_key: str = "SPX") -> Path:
    parts = []
    if banner:
        parts.append(f'<div class="warn">{banner}</div>')
    for key, (df, meta) in monitors.items():
        r = latest_reading(df, meta)
        parts.append(f"<section><h2>{html.escape(meta.get('label', key))}</h2>"
                     f"<p class='note'>Estimateur RV : {meta['rv_col']} — {html.escape(meta.get('rv_choice_reason', ''))}</p>"
                     f"{reading_cards(r)}{_fig_html(fig_iv_rv(df, meta))}{_fig_html(fig_spread(df, C.Z_THRESHOLD))}</section>")
    if study_key in monitors:
        df, meta = monitors[study_key]
        ft = fig_term(df)
        if ft is not None:
            parts.append(f"<section><h2>Structure par terme de la vol implicite</h2>{_fig_html(ft)}"
                         "<p class='note'>Contango (VIX &lt; VIX3M) : régime calme, le marché paie une prime pour l'incertitude future. "
                         "Backwardation : stress, la vol immédiate dépasse la vol à 3 mois.</p></section>")
        parts.append(f"<section><h2>Choix de l'estimateur de vol réalisée</h2>{_fig_html(fig_estimators(df))}"
                     "<p class='note'>Parkinson, Garman-Klass et Rogers-Satchell ignorent le gap de nuit et sous-estiment la vol totale ; "
                     "Yang-Zhang le capture et reste efficace, à condition que les prix d'ouverture soient fiables.</p></section>")
        parts.append(f"<section><h2>Prime de risque de variance</h2>{_fig_html(fig_vrp(df))}</section>")
    if snap is not None and not snap.empty:
        vix = monitors[study_key][0]["iv"] if study_key in monitors else None
        figs = fig_options(snap, term_snap if term_snap is not None else pd.DataFrame(), vix)
        parts.append("<section><h2>Chaînes d'options (snapshots quotidiens)</h2>"
                     + "".join(_fig_html(f) for f in figs)
                     + "<p class='note'>IV recalculée (Black-76) sur les mids OTM, forward implicite par parité call-put, "
                       "ATM 30j interpolé en variance totale, risk reversal 25Δ = IV(call 25Δ) − IV(put 25Δ).</p></section>")
    if study_key in monitors:
        df, meta = monitors[study_key]
        parts.append(f"<section><h2>Test du signal : un spread élevé précède-t-il une vente de vol gagnante ?</h2>"
                     f"{signal_study_html(df, meta, splits or ['2008-01-01', '2013-01-01', '2018-01-01'])}</section>")
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}"></script><style>{CSS}</style></head><body>
<header><div class="in"><div class="kicker">Implied vs realised volatility monitor</div><h1>{html.escape(title)}</h1>
<p>{subtitle}</p></div></header><main>{''.join(parts)}</main>
<footer>Généré le {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC · Kevin Gautier · github.com/kevingautierr-dev ·
Outil d'analyse, pas un conseil en investissement.</footer></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out


def build_report(out: Path | None = None) -> Path:
    """Live dashboard from data/processed (used by the daily job)."""
    monitors = {}
    summary = json.loads((C.PROCESSED / "latest.json").read_text()) if (C.PROCESSED / "latest.json").exists() else {"pairs": {}}
    for key, info in summary["pairs"].items():
        p = C.PROCESSED / f"monitor_{key}.parquet"
        if info.get("available") and p.exists():
            meta = {k: v for k, v in info.items() if k not in ("latest", "available")}
            monitors[key] = (pd.read_parquet(p), meta)
    missing = [f"{C.PAIRS[k]['label']} ({v.get('reason')})" for k, v in summary["pairs"].items() if not v.get("available")]
    banner = ("Séries non couvertes : " + "; ".join(missing)) if missing else ""
    snap = pd.read_csv(C.IV_SNAPSHOTS) if C.IV_SNAPSHOTS.exists() else None
    term = pd.read_csv(C.IV_TERM) if C.IV_TERM.exists() else None
    last = max((m[0].index.max() for m in monitors.values()), default=None)
    sub = f"La volatilité est-elle chère aujourd'hui ? Données au {last.date() if last is not None else 'n/d'}."
    return build_page(monitors, out or (C.DOCS / "index.html"), "La volatilité est-elle chère ?", sub,
                      snap, term, banner, splits=["2012-01-01", "2016-01-01", "2020-01-01", "2023-01-01"])
