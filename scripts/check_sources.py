"""Step 1 of the data plan: which series does Yahoo actually serve?

Writes docs/sources.md. Decision rule: if ^V2TX (VSTOXX) is missing or stale,
Europe is NOT covered for implied vol -> the project is presented as
"S&P 500 (+ Nasdaq-100)" and the CV line says so.

    python scripts/check_sources.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volmon.data import check_sources  # noqa: E402

if __name__ == "__main__":
    rep = check_sources()
    print(rep.to_string(index=False))
    sx = rep[rep["name"] == "SX5E_IV"]
    europe = bool(len(sx) and sx["ok"].iloc[0])
    verdict = ("VSTOXX disponible et à jour : la ligne Euro Stoxx 50 / VSTOXX est active."
               if europe else
               "VSTOXX absent ou périmé sur Yahoo : pas de vol implicite européenne gratuite. "
               "Périmètre affiché : S&P 500 / VIX (+ Nasdaq-100 / VXN). La vol réalisée Euro Stoxx 50 reste calculée.")
    out = ROOT / "docs" / "sources.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"# Vérification des sources\n\n{verdict}\n\n{rep.to_markdown(index=False)}\n", encoding="utf-8")
    print("\n" + verdict)
