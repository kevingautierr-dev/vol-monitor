"""Daily job: history -> option snapshots -> monitors -> docs/index.html.

    python scripts/daily.py              # full run
    python scripts/daily.py --no-options # skip option chains (e.g. weekends)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volmon.pipeline import run_daily  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-options", action="store_true")
    run_daily(skip_options=ap.parse_args().no_options)
