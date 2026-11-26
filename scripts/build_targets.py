"""Build validation targets for Experiment 2.

    python scripts/build_targets.py --exposure eloundou

Writes to data/processed/:
  indeed_targets.csv   monthly postings change vs pre-ChatGPT level per sector, with sector exposure
  ipums_targets.csv    quarterly employment and job-finding rates, age group x exposure quintile
                       (only when data/raw/ipums/cps.csv and the Census->SOC crosswalk are present)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.data.validation import indeed_targets, ipums_targets, load_cps, occupation_exposure_for_census  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure", default="eloundou", choices=["anthropic", "eloundou", "microsoft", "aioe"])
    a = ap.parse_args(argv)
    cfg = load_config()
    raw, proc = resolve(cfg["paths"]["raw"]), resolve(cfg["paths"]["processed"])
    occ = pd.read_csv(proc / "occupations.csv", dtype={"soc": str})

    post = raw / "indeed" / "job_postings_by_sector_US.csv"
    if post.exists():
        sm = pd.read_csv(Path(__file__).resolve().parent.parent / "labor_sim" / "data" / "indeed_sector_map.csv")
        t = indeed_targets(pd.read_csv(post), sm, occ, a.exposure)
        t.to_csv(proc / "indeed_targets.csv", index=False)
        last = t[t["month"] == t["month"].max()]
        print(f"Indeed: {t['display_name'].nunique()} sectors, latest month {t['month'].max():%Y-%m}")
        print("  log change in postings vs May-Nov 2022, by sector exposure tercile:")
        print(last.groupby("exposure_group", observed=True)["change_vs_pre"].mean().round(3).to_string())
    else:
        print("Indeed file missing: run scripts/download_data.py --only github")

    cps = load_cps(raw)
    xw_path = raw / "crosswalk" / "census_occ_to_soc.csv"
    if cps is not None and xw_path.exists():
        xw = pd.read_csv(xw_path, dtype={"soc": str})
        xw["soc"] = xw["soc"].str[:7]
        exp = occupation_exposure_for_census(occ, xw, a.exposure)
        t = ipums_targets(cps, exp, cfg["simulation"]["young_age_max"])
        t.to_csv(proc / "ipums_targets.csv", index=False)
        print(f"IPUMS: {len(t)} rows written")
    else:
        print("IPUMS targets skipped: need data/raw/ipums/cps.csv and data/raw/crosswalk/census_occ_to_soc.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
