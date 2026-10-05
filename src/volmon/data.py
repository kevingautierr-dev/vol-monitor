"""Data access: daily bars and vol indices (yfinance), option-chain snapshots,
and the source-availability check that decides whether Europe is covered."""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import config as C

NY = ZoneInfo("America/New_York")


def _yf():
    import yfinance as yf   # imported lazily so the analytics run without it
    return yf


# --------------------------------------------------------------------------- #
# Daily history
# --------------------------------------------------------------------------- #
def fetch_history(ticker: str, period: str = "max") -> pd.DataFrame:
    h = _yf().Ticker(ticker).history(period=period, auto_adjust=False, actions=False)
    if h.empty:
        return h
    h.index = pd.to_datetime(h.index.tz_localize(None).date)
    h = h[["Open", "High", "Low", "Close"]].astype(float)
    return h[~h.index.duplicated(keep="last")].dropna()


def completed_bars(h: pd.DataFrame, now: dt.datetime | None = None) -> pd.DataFrame:
    """Drop today's bar if the US cash session is still open (run mid-day)."""
    now = now or dt.datetime.now(NY)
    if len(h) and h.index[-1].date() == now.date() and now.time() < dt.time(16, 15):
        return h.iloc[:-1]
    return h


def refresh_history() -> dict[str, pd.DataFrame]:
    """Download every series in the config and store it as Parquet."""
    C.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    tickers = {}
    for key, p in C.PAIRS.items():
        tickers[key] = p["underlying"]
        tickers[key + "_IV"] = p["iv_index"]
    tickers.update(C.TERM_TICKERS)
    tickers.update(C.EXTRA_TICKERS)
    out = {}
    for name, t in tickers.items():
        try:
            h = completed_bars(fetch_history(t))
        except Exception as e:     # one bad ticker must not kill the run
            print(f"[warn] {t}: {e}")
            continue
        if h.empty:
            print(f"[warn] {t}: no data")
            continue
        h.to_parquet(C.HISTORY_DIR / f"{name}.parquet")
        out[name] = h
    return out


def load_history(name: str) -> pd.DataFrame:
    p = C.HISTORY_DIR / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Source check (step 1 of the data plan)
# --------------------------------------------------------------------------- #
def check_sources(stale_days: int = 5) -> pd.DataFrame:
    """Is each series available, how long, and is it fresh?"""
    names = {**{k: v["underlying"] for k, v in C.PAIRS.items()},
             **{k + "_IV": v["iv_index"] for k, v in C.PAIRS.items()},
             **C.TERM_TICKERS, **C.EXTRA_TICKERS}
    rows = []
    today = pd.Timestamp.today().normalize()
    for name, t in names.items():
        try:
            h = fetch_history(t)
        except Exception as e:
            rows.append({"name": name, "ticker": t, "ok": False, "error": str(e)})
            continue
        if h.empty:
            rows.append({"name": name, "ticker": t, "ok": False, "error": "empty"})
            continue
        last = h.index.max()
        bdays_old = int(np.busday_count(last.date(), today.date()))
        rows.append({
            "name": name, "ticker": t, "first": str(h.index.min().date()), "last": str(last.date()),
            "n": len(h), "business_days_old": bdays_old, "ok": bdays_old <= stale_days,
            "share_open_eq_prev_close": float(np.isclose(h["Open"], h["Close"].shift()).mean()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Option-chain snapshots
# --------------------------------------------------------------------------- #
def _expiry_close(expiry: str) -> dt.datetime:
    d = dt.date.fromisoformat(expiry)
    return dt.datetime.combine(d, dt.time(16, 0), tzinfo=NY)


def year_fraction(expiry: str, asof: dt.datetime) -> float:
    return (_expiry_close(expiry) - asof).total_seconds() / (365.0 * 86400)


def snapshot_chain(ticker: str, asof: dt.datetime | None = None) -> tuple[pd.DataFrame, float]:
    """All expiries within [MIN_DTE, MAX_DTE], long format, one row per option."""
    yf = _yf()
    asof = asof or dt.datetime.now(NY)
    tk = yf.Ticker(ticker)
    spot = float(tk.history(period="5d")["Close"].iloc[-1])
    frames = []
    for exp in tk.options:
        T = year_fraction(exp, asof)
        if not (C.MIN_DTE / 365 <= T <= C.MAX_DTE / 365):
            continue
        try:
            ch = tk.option_chain(exp)
        except Exception as e:
            print(f"[warn] {ticker} {exp}: {e}")
            continue
        for side, t in ((ch.calls, "C"), (ch.puts, "P")):
            if side is None or side.empty:
                continue
            f = side[["strike", "bid", "ask", "lastPrice", "volume", "openInterest", "impliedVolatility"]].copy()
            f["type"] = t
            f["expiry"] = exp
            f["T"] = T
            frames.append(f)
    chain = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not chain.empty:
        chain["asof"] = asof.isoformat(timespec="minutes")
        chain["spot"] = spot
    return chain, spot


def risk_free_rate(default: float = 0.04) -> float:
    """13-week T-bill yield (^IRX, in %) as a flat rate. The forward is taken
    from put-call parity, so the rate only enters through the discount factor."""
    h = load_history("IRX")
    if h.empty:
        try:
            h = fetch_history("^IRX", "1mo")
        except Exception:
            return default
    return float(h["Close"].iloc[-1]) / 100 if not h.empty else default
