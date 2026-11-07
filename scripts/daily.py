"""Daily job: history -> option snapshots -> monitors -> docs/index.html.

    python scripts/daily.py                 # full run; option chains only if the US session is open
    python scripts/daily.py --no-options    # never snapshot option chains
    python scripts/daily.py --force-options # snapshot option chains whatever the time
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volmon.pipeline import run_daily  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-options", action="store_true")
    g.add_argument("--force-options", action="store_true")
    a = ap.parse_args()
    run_daily(skip_options=a.no_options, force_options=a.force_options)
