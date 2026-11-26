"""Download the open datasets into data/raw/.

Usage:
    python scripts/download_data.py            # everything that can be scripted
    python scripts/download_data.py --only github
    python scripts/download_data.py --only oews onet anthropic

Pew (Wave 157) and IPUMS CPS need a free account and are downloaded by hand;
see data/README.md for the exact steps and where to put the files.

Every download is recorded in data/raw/MANIFEST.csv (source, URL, date, size)
so the report can state exactly which version was used.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# BLS rejects requests without a descriptive User-Agent. Put your e-mail here.
USER_AGENT = "labor-sim graduation project (contact: your-email@example.edu)"

GH = "https://raw.githubusercontent.com"
GITHUB_FILES = {
    # Eloundou et al. (2024): occupation-level exposure + O*NET tasks (MIT)
    "eloundou/occ_level.csv": f"{GH}/openai/GPTs-are-GPTs/main/data/occ_level.csv",
    "eloundou/full_onet_data.tsv": f"{GH}/openai/GPTs-are-GPTs/main/data/full_onet_data.tsv",
    "eloundou/national_May2021_dl.csv": f"{GH}/openai/GPTs-are-GPTs/main/data/national_May2021_dl.csv",
    # Tomlinson et al. (2025), Microsoft Working with AI (CC BY 4.0)
    "microsoft/ai_applicability_scores.csv": f"{GH}/microsoft/working-with-ai/main/ai_applicability_scores.csv",
    "microsoft/soc_metrics.csv": f"{GH}/microsoft/working-with-ai/main/soc_metrics.csv",
    # Felten, Raj & Seamans (2021), AIOE
    "aioe/AIOE_DataAppendix.xlsx": f"{GH}/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx",
    "aioe/Language Modeling AIOE and AIIE.xlsx": f"{GH}/AIOE-Data/AIOE/main/Language%20Modeling%20AIOE%20and%20AIIE.xlsx",
    # Indeed Hiring Lab (CC BY 4.0)
    "indeed/job_postings_by_sector_US.csv": f"{GH}/hiring-lab/job_postings_tracker/master/US/job_postings_by_sector_US.csv",
    "indeed/aggregate_job_postings_US.csv": f"{GH}/hiring-lab/job_postings_tracker/master/US/aggregate_job_postings_US.csv",
    "indeed/AI_posting.csv": f"{GH}/hiring-lab/ai-tracker/main/AI_posting.csv",
}

# BLS OEWS national files, May of each year. Check the URL pattern on
# https://www.bls.gov/oes/tables.htm if a year fails.
OEWS_YEARS = [2022, 2023, 2024, 2025]
OEWS_URL = "https://www.bls.gov/oes/special-requests/oesm{yy}nat.zip"

# O*NET text database. Check the current version on
# https://www.onetcenter.org/database.html and update if needed.
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_31_0_text.zip"

ANTHROPIC_REPO = "Anthropic/EconomicIndex"

# BLS SOC 2010 -> 2018 crosswalk (needed because AIOE uses SOC 2010 codes)
CROSSWALK_URL = "https://www.bls.gov/soc/2018/soc_2010_to_2018_crosswalk.xlsx"


def _log(name: str, url: str, path: Path) -> None:
    manifest = RAW / "MANIFEST.csv"
    new = not manifest.exists()
    with manifest.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "url", "downloaded_utc", "bytes"])
        w.writerow([name, url, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), path.stat().st_size])


def fetch(url: str, dest: Path, headers: dict | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  GET {url}")
    r = requests.get(url, headers=headers or {}, timeout=300)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_github() -> None:
    print("GitHub datasets")
    for name, url in GITHUB_FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"  skip {name} (exists)")
            continue
        fetch(url, dest)
        _log(name, url, dest)


def download_oews() -> None:
    print("BLS OEWS national")
    for year in OEWS_YEARS:
        url = OEWS_URL.format(yy=str(year)[2:])
        out_dir = RAW / "oews" / str(year)
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  skip {year} (exists)")
            continue
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=300)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ! {year}: {exc}. Download it by hand from https://www.bls.gov/oes/tables.htm")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(r.content)).extractall(out_dir)
        _log(f"oews/{year}", url, next(out_dir.rglob("*.xlsx")))


def download_onet() -> None:
    print("O*NET database")
    out_dir = RAW / "onet"
    if out_dir.exists() and any(out_dir.rglob("Task Statements.txt")):
        print("  skip (exists)")
        return
    r = requests.get(ONET_URL, timeout=600)
    r.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(out_dir)
    _log("onet", ONET_URL, next(out_dir.rglob("Task Statements.txt")))


def download_anthropic() -> None:
    print("Anthropic Economic Index (Hugging Face)")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  ! pip install huggingface_hub, or download by hand from "
              f"https://huggingface.co/datasets/{ANTHROPIC_REPO}")
        return
    out_dir = RAW / "anthropic"
    snapshot_download(repo_id=ANTHROPIC_REPO, repo_type="dataset", local_dir=out_dir)
    for p in out_dir.rglob("*.csv"):
        _log(f"anthropic/{p.relative_to(out_dir)}", f"hf://{ANTHROPIC_REPO}", p)


def download_crosswalk() -> None:
    print("BLS SOC 2010 -> 2018 crosswalk")
    dest = RAW / "crosswalk" / "soc_2010_to_2018_crosswalk.xlsx"
    if dest.exists():
        print("  skip (exists)")
        return
    try:
        fetch(CROSSWALK_URL, dest, headers={"User-Agent": USER_AGENT})
        _log("crosswalk", CROSSWALK_URL, dest)
    except requests.RequestException as exc:
        print(f"  ! {exc}. Download it by hand from https://www.bls.gov/soc/2018/crosswalks_used_by_agencies.htm")


STEPS = {"github": download_github, "crosswalk": download_crosswalk, "oews": download_oews,
         "onet": download_onet, "anthropic": download_anthropic}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(STEPS), help="run only these steps")
    args = ap.parse_args(argv)
    for name in args.only or list(STEPS):
        STEPS[name]()
    print("\nManual steps left: Pew Wave 157 and IPUMS CPS (see data/README.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
