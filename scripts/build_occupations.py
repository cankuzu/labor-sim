"""Build data/processed/occupations.csv from whatever is in data/raw/.

    python scripts/build_occupations.py

Writes occupations.csv and occupations_report.json (coverage and unmatched SOC codes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.data.occupations import build_occupations  # noqa: E402


def main() -> int:
    cfg = load_config()
    raw, out = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    print("Building occupation table")
    occ, report = build_occupations(raw)
    occ.to_csv(out / "occupations.csv", index=False)
    (out / "occupations_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {len(occ)} occupations to {out / 'occupations.csv'}")
    print("Exposure coverage:", report["coverage"])
    if report.get("emp_years") and 2025 not in report["emp_years"]:
        print("Note: OEWS 2022-2025 not found yet; using", report["emp_years"], "(dev only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
