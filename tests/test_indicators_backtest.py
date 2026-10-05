import numpy as np
import pandas as pd
import pytest

from volmon import backtest as bt
from volmon import indicators as ind
from volmon import pipeline
from volmon.simulate import simulate_ohlc
from test_implied import make_chain


def test_iv_percentile_and_rank():
    x = pd.Series(np.arange(300, dtype=float))
    assert ind.iv_percentile(x, 252).iloc[-1] == 1.0       # today above all past days
    assert ind.iv_rank(x, 252).iloc[-1] == 1.0
    y = pd.Series(np.r_[np.arange(260.0), [10.0]])
    assert ind.iv_percentile(y, 252).iloc[-1] == pytest.approx(2 / 252)   # past window = 8..259, only 8 and 9 < 10
    assert ind.iv_percentile(x, 252).iloc[:252].isna().all()


def test_signal_requires_contango_for_rich():
    df = pd.DataFrame({"spread_z": [2.0, 2.0, -2.0, -2.0, 0.0, np.nan],
                       "contango": [True, False, False, True, True, True]})
    s = ind.signal(df, 1.0, use_term=True)
    assert s.tolist() == [1, 0, -1, 0, 0, 0]
    assert ind.signal(df, 1.0, use_term=False).tolist() == [1, 1, -1, -1, 0, 0]


def test_short_variance_pnl_formula():
    df = pd.DataFrame({"iv": [0.20, 0.20], "fwd_rv": [0.15, 0.40]})
    p = bt.short_variance_pnl(df, cost=0.0)
    assert p.iloc[0] == pytest.approx((0.04 - 0.0225) / 0.4 * 100)    # +4.375 pts
    assert p.iloc[1] == pytest.approx((0.04 - 0.16) / 0.4 * 100)      # -30 pts: convexity
    assert p.iloc[1] < -6 * p.iloc[0]


def test_newey_west_matches_iid_tstat_at_lag0():
    x = pd.Series(np.random.default_rng(0).normal(0.1, 1, 2000))
    t_iid = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert bt.newey_west_tstat(x, 0) == pytest.approx(t_iid)


def test_newey_west_deflates_overlap():
    rng = np.random.default_rng(1)
    e = rng.normal(0, 1, 5000)
    overlapping = pd.Series(np.convolve(e, np.ones(21), "valid")) + 2   # MA(20)
    naive = overlapping.mean() / (overlapping.std() / np.sqrt(len(overlapping)))
    assert bt.newey_west_tstat(overlapping, 20) < naive / 2


def test_monitor_on_synthetic_has_no_lookahead_in_signal():
    ohlc = simulate_ohlc(n_days=900, sigma=0.18, overnight_share=0.2, seed=3)
    iv = pd.Series(0.20, index=ohlc.index)
    df, meta = ind.build_monitor(ohlc, iv)
    assert meta["rv_col"] == "yz_21"
    t = 600
    shocked = ohlc.copy(); shocked.iloc[t + 1:] *= np.linspace(1, 3, len(shocked) - t - 1)[:, None]
    df2, _ = ind.build_monitor(shocked, iv)
    cols = ["rv", "spread", "spread_z", "iv_rank_1y", "signal"]
    pd.testing.assert_frame_equal(df.iloc[: t + 1][cols], df2.iloc[: t + 1][cols])
    # but the ex-post VRP does change: it is allowed to look ahead
    assert not np.allclose(df["vrp_var_expost"].iloc[t - 5:t], df2["vrp_var_expost"].iloc[t - 5:t])


def test_choose_rv_falls_back_when_opens_are_copied():
    ohlc = simulate_ohlc(n_days=300, seed=4)
    ohlc["Open"] = ohlc["Close"].shift(1)
    ohlc = ohlc.dropna()
    ohlc["High"] = ohlc[["High", "Open"]].max(axis=1)
    ohlc["Low"] = ohlc[["Low", "Open"]].min(axis=1)
    col, why = ind.choose_rv(ohlc)
    assert col == "cc_21" and "close-to-close" in why


def test_chain_metrics_long_format():
    frames = []
    for d, exp in ((16, "2026-10-21"), (44, "2026-11-18")):
        ch, _ = make_chain(T=d / 365, atm=0.16 + 0.001 * d)
        ch["T"] = d / 365
        ch["expiry"] = exp
        frames.append(ch)
    term, cm = pipeline.chain_metrics(pd.concat(frames), spot=100.0, r=0.04)
    assert list(term["expiry"]) == ["2026-10-21", "2026-11-18"]
    assert 0.176 < cm["atm_iv_30d"] < 0.204
    assert cm["rr25_30d"] < 0
