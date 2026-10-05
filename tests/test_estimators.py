import numpy as np
import pytest

from volmon import estimators as est
from volmon.simulate import simulate_ohlc

SIGMA = 0.20


@pytest.fixture(scope="module")
def no_gap():
    return simulate_ohlc(n_days=3000, sigma=SIGMA, overnight_share=0.0, seed=1)


@pytest.fixture(scope="module")
def with_gap():
    return simulate_ohlc(n_days=3000, sigma=SIGMA, overnight_share=0.25, seed=2)


@pytest.mark.parametrize("name", list(est.ESTIMATORS))
def test_recovers_sigma_without_gap(no_gap, name):
    rv = est.ESTIMATORS[name](no_gap, 63).dropna()
    # discrete monitoring biases range estimators slightly down -> 6 % tolerance
    assert rv.mean() == pytest.approx(SIGMA, rel=0.06)


def test_range_estimators_miss_overnight_gap(with_gap):
    """With 25 % of variance overnight, Parkinson/GK/RS only see ~sqrt(0.75)."""
    intraday_only = SIGMA * np.sqrt(0.75)
    for f in (est.parkinson, est.garman_klass, est.rogers_satchell):
        assert f(with_gap, 63).dropna().mean() == pytest.approx(intraday_only, rel=0.06)
    # close-to-close and Yang-Zhang capture the full variance
    assert est.close_to_close(with_gap, 63).dropna().mean() == pytest.approx(SIGMA, rel=0.05)
    assert est.yang_zhang(with_gap, 63).dropna().mean() == pytest.approx(SIGMA, rel=0.06)


def test_range_estimators_are_more_efficient(no_gap):
    """Lower dispersion of the estimate around the true sigma than close-to-close."""
    sd_cc = est.close_to_close(no_gap, 21).dropna().std()
    sd_pk = est.parkinson(no_gap, 21).dropna().std()
    sd_yz = est.yang_zhang(no_gap, 21).dropna().std()
    assert sd_pk < 0.7 * sd_cc
    assert sd_yz < sd_cc


def test_no_lookahead(no_gap):
    """Changing a future bar must not change today's estimate."""
    t = 500
    base = est.yang_zhang(no_gap, 21).iloc[t]
    shocked = no_gap.copy()
    shocked.iloc[t + 1:, :] *= 1.5
    assert est.yang_zhang(shocked, 21).iloc[t] == pytest.approx(base)


def test_forward_realised_var_alignment(no_gap):
    fwd = est.forward_realised_var(no_gap["Close"], 21)
    r = np.log(no_gap["Close"]).diff()
    t = 100
    manual = (r.iloc[t + 1:t + 22] ** 2).mean() * 252
    assert fwd.iloc[t] == pytest.approx(manual)
    assert fwd.iloc[-21:].isna().all()


def test_data_quality_flags_open_equal_prev_close(no_gap):
    df = no_gap.copy()
    df["Open"] = df["Close"].shift(1)
    df = df.dropna()
    df["High"] = df[["High", "Open"]].max(axis=1)
    df["Low"] = df[["Low", "Open"]].min(axis=1)
    q = est.data_quality(df)
    assert q["share_open_eq_prev_close"] > 0.99
