import numpy as np
import pandas as pd
import pytest

from volmon import implied as iv


def smile_vol(k, atm=0.18, skew=-0.35, curv=0.6):
    return atm + skew * k + curv * k ** 2


def make_chain(spot=100.0, T=30 / 365, r=0.04, q=0.015, half_spread=0.005, **kw):
    F = spot * np.exp((r - q) * T)
    df = np.exp(-r * T)
    strikes = np.arange(70, 131, 1.0)
    rows = []
    for K in strikes:
        s = smile_vol(np.log(K / F), **kw)
        for t in ("C", "P"):
            p = float(iv.black76(F, K, T, df, s, t == "C"))
            if p < 0.01:
                continue
            rows.append({"strike": K, "type": t, "bid": p * (1 - half_spread), "ask": p * (1 + half_spread)})
    return pd.DataFrame(rows), F


@pytest.mark.parametrize("is_call", [True, False])
def test_black76_iv_roundtrip(is_call):
    F, K, T, df = 100.0, 95.0, 0.25, 0.99
    for s in (0.05, 0.2, 0.8):
        p = float(iv.black76(F, K, T, df, s, is_call))
        assert iv.implied_vol(p, F, K, T, df, is_call) == pytest.approx(s, abs=1e-6)


def test_iv_nan_below_intrinsic():
    assert np.isnan(iv.implied_vol(1.0, 100, 90, 0.5, 1.0, True))


def test_implied_forward_from_parity():
    chain, F = make_chain()
    q = iv.clean_quotes(chain)
    est_F = iv.implied_forward(q, np.exp(-0.04 * 30 / 365), spot=100.0)
    assert est_F == pytest.approx(F, abs=0.01)


def test_smile_metrics_recover_atm_and_negative_skew():
    T = 30 / 365
    chain, F = make_chain(T=T)
    sm = iv.expiry_smile(chain, T, 0.04, 100.0)
    m = iv.smile_metrics(sm)
    assert m["atm_iv"] == pytest.approx(0.18, abs=2e-3)
    assert m["rr25"] < 0          # equity skew: puts richer than calls
    assert m["iv_25p"] > m["atm_iv"] > m["iv_25c"]
    # 25d strikes sit roughly at k = -/+ 0.674*sigma*sqrt(T)
    k25 = 0.674 * 0.18 * np.sqrt(T)
    assert m["rr25"] == pytest.approx(smile_vol(k25) - smile_vol(-k25), abs=0.01)


def test_constant_maturity_total_variance():
    term = pd.DataFrame({"T": [20 / 365, 40 / 365], "atm_iv": [0.15, 0.20],
                         "rr25": [-0.04, -0.03], "bf25": [0.004, 0.003]})
    cm = iv.constant_maturity(term, 30)
    expected = np.sqrt(((0.15 ** 2 * 20 + 0.20 ** 2 * 40) / 2) / 30)
    assert cm["atm_iv_30d"] == pytest.approx(expected)
    assert cm["rr25_30d"] == pytest.approx(-0.035)


def test_term_structure_end_to_end():
    chains = {}
    for d in (9, 23, 37, 65):
        T = d / 365
        chains[T], _ = make_chain(T=T, atm=0.15 + 0.001 * d)
    term = iv.term_structure(chains, 0.04, 100.0)
    assert len(term) == 4
    assert term["atm_iv"].is_monotonic_increasing      # contango by construction
    cm = iv.constant_maturity(term)
    assert 0.15 + 0.023 < cm["atm_iv_30d"] < 0.15 + 0.037
