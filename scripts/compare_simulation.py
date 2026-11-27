"""Experiment 2: does the simulation reproduce the young x high-exposure pattern seen in the CPS?

    python scripts/compare_simulation.py                       # every run in results/simulations/
    python scripts/compare_simulation.py --runs results/simulations/*rule*

For each run (metrics.csv) and for the IPUMS targets (data/processed/ipums_targets.csv) it computes
the difference-in-differences on the employment rate

    DiD(t) = [young Q5 - young Q1] - [older Q5 - older Q1]

and its change from the start (2022Q4) to the 2025 average. Negative = young workers in the most
exposed occupations lose ground relative to everyone else. Runs are grouped by policy/model and
summarised as mean and spread across seeds, so seed noise is visible next to the CPS value.
Rates are pooled over each age group x quintile cell by headcount (CPS: by n).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402

START, POST = "2022Q4", ("2025Q1", "2025Q2", "2025Q3", "2025Q4")


def did_series(m: pd.DataFrame, rate: str = "employment_rate", q: str = "exposure_quintile") -> pd.Series:
    young = sorted(m["age_group"].unique())[0]          # "22-25" sorts before "26+"
    p = m.pivot_table(index="quarter", columns=["age_group", q], values=rate)
    older = [a for a in m["age_group"].unique() if a != young][0]
    return (p[(young, 5)] - p[(young, 1)]) - (p[(older, 5)] - p[(older, 1)])


def summarise(d: pd.Series) -> dict:
    post = d[d.index.isin(POST)].mean()
    return {"did_start": d.get(START), "did_2025": post, "change": post - d.get(START)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path)
    a = ap.parse_args(argv)
    cfg = load_config()
    runs = a.runs or sorted(resolve(cfg["paths"]["results"]).glob("simulations/*"))
    rows = []
    for r in runs:
        f = Path(r) / "metrics.csv"
        if not f.exists():
            continue
        name = Path(r).name                     # <time>_<tag>_<exposure>_s<seed>
        parts = name.split("_")
        m = pd.read_csv(f)
        young = m[(m["age_group"] == "22-25") & (m["quarter"] == START)]["labor_force"].sum()
        rows.append({"run": name, "policy": parts[1], "exposure": parts[2], "seed": parts[-1],
                     "young_workers": int(young)} | summarise(did_series(m)))
    sim = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    if not sim.empty:
        print("Simulation runs, DiD on employment rate (points):")
        g = sim.groupby(["policy", "exposure"])
        out = g["change"].agg(["count", "mean", "std", "min", "max"]).join(g["young_workers"].mean().rename("young_n"))
        print((out * [1, 100, 100, 100, 100, 1]).round(2).to_string())
    tgt = resolve(cfg["paths"]["processed"]) / "ipums_targets.csv"
    if tgt.exists():
        t = pd.read_csv(tgt)
        s = summarise(did_series(t, q="quintile"))
        print(f"\nCPS (IPUMS) target: DiD {START} {100 * s['did_start']:.2f}, 2025 avg {100 * s['did_2025']:.2f}, "
              f"change {100 * s['change']:.2f} points")
    else:
        print(f"\nNo IPUMS target table at {tgt} (run scripts/build_targets.py on the machine with the CPS file).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
