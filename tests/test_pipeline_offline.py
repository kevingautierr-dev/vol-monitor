"""End-to-end run of the daily job with Yahoo replaced by synthetic data:
history -> option snapshots -> monitors -> HTML dashboard."""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from volmon import config as C
from volmon import data as D
from volmon import pipeline
from volmon.simulate import simulate_ohlc
from test_implied import make_chain

N = 900


def _index():
    return pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=N)


def fake_history(ticker, period="max"):
    idx = _index()
    if ticker in ("^GSPC", "^NDX", "^STOXX50E", "^FCHI"):
        h = simulate_ohlc(n_days=N, sigma=0.17, overnight_share=0.2, seed=hash(ticker) % 1000)
        h.index = idx
        return h
    if ticker == "^V2TX":                      # simulate a dead European source
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])
    rng = np.random.default_rng(abs(hash(ticker)) % 1000)
    base = {"^VIX9D": 17, "^VIX": 19, "^VIX3M": 21, "^VIX6M": 22, "^VVIX": 90, "^VXN": 23, "^IRX": 4.0}[ticker]
    lvl = base * np.exp(np.cumsum(rng.normal(0, 0.03, N)) * 0.3)
    return pd.DataFrame({"Open": lvl, "High": lvl * 1.02, "Low": lvl * 0.98, "Close": lvl}, index=idx)


def fake_snapshot(ticker, asof=None):
    frames = []
    for d in (7, 21, 35, 63, 120):
        ch, _ = make_chain(T=d / 365, atm=0.15 + 0.0004 * d)
        ch["T"] = d / 365
        ch["expiry"] = (dt.date.today() + dt.timedelta(days=d)).isoformat()
        for c in ("lastPrice", "volume", "openInterest", "impliedVolatility"):
            ch[c] = np.nan
        frames.append(ch)
    chain = pd.concat(frames, ignore_index=True)
    chain["asof"] = "x"; chain["spot"] = 100.0
    return chain, 100.0


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name in ("HISTORY_DIR", "CHAINS_DIR", "PROCESSED", "DOCS"):
        monkeypatch.setattr(C, name, tmp_path / name.lower())
    monkeypatch.setattr(C, "IV_SNAPSHOTS", tmp_path / "iv_snapshots.csv")
    monkeypatch.setattr(C, "IV_TERM", tmp_path / "iv_term.csv")
    monkeypatch.setattr(D, "fetch_history", fake_history)
    monkeypatch.setattr(D, "snapshot_chain", fake_snapshot)
    return tmp_path


def test_run_daily_end_to_end(sandbox):
    pipeline.run_daily()
    summary = json.loads((C.PROCESSED / "latest.json").read_text())
    assert summary["pairs"]["SPX"]["available"] and summary["pairs"]["SPX"]["term_filter"]
    assert summary["pairs"]["NDX"]["available"]
    assert not summary["pairs"]["SX5E"]["available"]          # Europe not covered -> flagged
    snaps = pd.read_csv(C.IV_SNAPSHOTS)
    assert set(snaps["ticker"]) == set(C.OPTION_TICKERS)
    assert (snaps["atm_iv_30d"].between(0.15, 0.17)).all()
    assert (snaps["rr25_30d"] < 0).all()
    html = (C.DOCS / "index.html").read_text()
    for needle in ("Structure par terme", "Chaînes d'options", "Test du signal", "Euro Stoxx 50 / VSTOXX"):
        assert needle in html
    # idempotent: same day twice does not duplicate rows
    pipeline.snapshot_all()
    assert len(pd.read_csv(C.IV_SNAPSHOTS)) == len(snaps)
