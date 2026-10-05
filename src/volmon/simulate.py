"""Synthetic OHLC paths with known volatility, used by the unit tests to check
that each estimator recovers the true sigma (and fails where theory says it
should, e.g. range estimators miss the overnight gap)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import TRADING_DAYS


def simulate_ohlc(
    n_days: int = 2000,
    sigma: float = 0.20,
    overnight_share: float = 0.0,
    steps_per_day: int = 500,
    drift: float = 0.0,
    s0: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """GBM with an optional overnight gap.

    ``overnight_share`` is the fraction of the daily variance realised between
    close and next open (typically 15-25 % for US equity indices).
    """
    rng = np.random.default_rng(seed)
    dvar = sigma ** 2 / TRADING_DAYS
    var_night = overnight_share * dvar
    var_day = (1 - overnight_share) * dvar
    mu_step = drift / TRADING_DAYS / steps_per_day

    rows = []
    close_prev = s0
    for _ in range(n_days):
        gap = rng.normal(-0.5 * var_night, np.sqrt(var_night)) if var_night > 0 else 0.0
        o = close_prev * np.exp(gap)
        incr = rng.normal(mu_step - 0.5 * var_day / steps_per_day,
                          np.sqrt(var_day / steps_per_day), steps_per_day)
        path = o * np.exp(np.concatenate([[0.0], np.cumsum(incr)]))
        rows.append((o, path.max(), path.min(), path[-1]))
        close_prev = path[-1]

    idx = pd.bdate_range("2010-01-04", periods=n_days)
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])
