"""Daily job and monitor assembly.

``run_daily`` = refresh history -> snapshot option chains -> compute IV
metrics -> append to CSV -> rebuild monitors -> write the static dashboard.
Each step is isolated so a failure in one (e.g. Yahoo option endpoint down)
does not lose the others.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import implied as IV
from . import indicators as ind


# --------------------------------------------------------------------------- #
# Option snapshot -> IV metrics
# --------------------------------------------------------------------------- #
def chain_metrics(chain: pd.DataFrame, spot: float, r: float) -> tuple[pd.DataFrame, dict]:
    chains = {T: g for T, g in chain.groupby("T")}
    term = IV.term_structure(chains, r, spot)
    if term.empty:
        return term, {}
    expiry_by_T = chain.groupby("T")["expiry"].first()
    term["expiry"] = term["T"].map(expiry_by_T)
    term["dte"] = (term["T"] * 365).round(1)
    cm = IV.constant_maturity(term, IV.TARGET_DAYS)
    return term, cm


def snapshot_all(asof: dt.datetime | None = None) -> list[dict]:
    asof = asof or dt.datetime.now(D.NY)
    day = asof.date().isoformat()
    r = D.risk_free_rate()
    rows = []
    for t in C.OPTION_TICKERS:
        try:
            chain, spot = D.snapshot_chain(t, asof)
        except Exception as e:
            print(f"[warn] snapshot {t}: {e}")
            continue
        if chain.empty:
            print(f"[warn] snapshot {t}: empty chain")
            continue
        out_dir = C.CHAINS_DIR / t
        out_dir.mkdir(parents=True, exist_ok=True)
        chain.to_parquet(out_dir / f"{day}.parquet", compression="zstd")

        term, cm = chain_metrics(chain, spot, r)
        if not cm:
            continue
        row = {"date": day, "ticker": t, "spot": spot, "r": r, "n_expiries": len(term), **cm}
        rows.append(row)
        term.insert(0, "ticker", t)
        term.insert(0, "date", day)
        _append_csv(C.IV_TERM, term, keys=["date", "ticker", "expiry"])
    if rows:
        _append_csv(C.IV_SNAPSHOTS, pd.DataFrame(rows), keys=["date", "ticker"])
    return rows


def _append_csv(path, new: pd.DataFrame, keys: list[str]):
    """Idempotent append: re-running the same day replaces that day's rows."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=keys, keep="last")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6g")


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #
def term_frame() -> pd.DataFrame | None:
    cols = {}
    for name in ("VIX9D", "VIX", "VIX3M", "VIX6M"):
        h = D.load_history(name)
        if not h.empty:
            cols[name] = h["Close"] / 100
    return pd.DataFrame(cols) if {"VIX", "VIX3M"} <= set(cols) else None


def build_all_monitors(min_fresh_days: int = 10) -> dict:
    """One monitor per index pair whose IV index is available and fresh."""
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    term = term_frame()
    summary = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "pairs": {}}
    for key, p in C.PAIRS.items():
        und, ivh = D.load_history(key), D.load_history(key + "_IV")
        if und.empty or ivh.empty:
            summary["pairs"][key] = {"available": False, "reason": "missing data"}
            continue
        age = np.busday_count(ivh.index.max().date(), dt.date.today())
        if age > min_fresh_days:
            summary["pairs"][key] = {"available": False, "reason": f"IV index stale ({age} business days)"}
            continue
        df, meta = ind.build_monitor(und, ivh["Close"] / 100, term if key == "SPX" else None)
        df.to_parquet(C.PROCESSED / f"monitor_{key}.parquet")
        summary["pairs"][key] = {"available": True, "label": p["label"], **meta,
                                 "latest": ind.latest_reading(df, meta)}
    (C.PROCESSED / "latest.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_daily(skip_options: bool = False):
    print("1/4 history"); D.refresh_history()
    if not skip_options:
        print("2/4 option snapshots"); print(snapshot_all())
    print("3/4 monitors"); summary = build_all_monitors()
    print(json.dumps(summary, indent=2, default=str))
    print("4/4 dashboard")
    from .report import build_report
    build_report()
