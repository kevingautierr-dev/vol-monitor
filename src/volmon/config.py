"""Universe and parameters. Everything the daily job touches is defined here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
HISTORY_DIR = DATA / "history"          # daily bars, rebuilt each run
CHAINS_DIR = DATA / "chains"            # raw option-chain snapshots (append only)
IV_SNAPSHOTS = DATA / "iv_snapshots.csv"
IV_TERM = DATA / "iv_term.csv"
PROCESSED = DATA / "processed"
DOCS = ROOT / "docs"                    # static dashboard (GitHub Pages)

# Underlying index -> its 30-day implied vol index. The VSTOXX line is kept only
# if the source check finds fresh data on Yahoo (see scripts/check_sources.py).
PAIRS = {
    "SPX": {"underlying": "^GSPC", "iv_index": "^VIX", "label": "S&P 500 / VIX"},
    "NDX": {"underlying": "^NDX", "iv_index": "^VXN", "label": "Nasdaq-100 / VXN"},
    "SX5E": {"underlying": "^STOXX50E", "iv_index": "^V2TX", "label": "Euro Stoxx 50 / VSTOXX"},
}

# VIX term structure (9 days, 30 days, 3 months, 6 months) + vol of vol
TERM_TICKERS = {"VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M", "VVIX": "^VVIX"}

EXTRA_TICKERS = {"IRX": "^IRX", "FCHI": "^FCHI"}   # 13-week T-bill (rate), CAC 40 (RV only)

# Option chains snapshotted every day (Yahoo only serves US listed options)
OPTION_TICKERS = ["SPY", "QQQ"]
MAX_DTE = 400          # ignore LEAPS beyond this to keep snapshots small
MIN_DTE = 2

# Monitor parameters
RV_WINDOWS = (10, 21, 63)
RV_FOR_SPREAD = "yz_21"          # falls back to cc_21 if open prices are unreliable
ZSCORE_WINDOW = 252
Z_THRESHOLD = 1.0
HORIZON = 21                      # trading days ~ 30 calendar days (VIX horizon)
COST_VOL_PTS = 0.5                # round-trip cost charged to each variance sale
