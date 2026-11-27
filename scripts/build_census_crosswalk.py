"""Turn the Census Bureau's 2018 occupation code list into data/raw/crosswalk/census_occ_to_soc.csv.

    python scripts/build_census_crosswalk.py
    python scripts/build_census_crosswalk.py --xlsx ~/Downloads/2018-occupation-code-list-and-crosswalk.xlsx

The workbook lists each 4-digit Census code with its 2018 SOC code. Some SOC codes are
aggregates written with X (e.g. 13-20XX) or as broad groups ending in 0 (e.g. 11-2030); these are expanded to every detailed SOC
in data/processed/occupations.csv that starts with the non-X part. The sheet layout differs
between releases, so every sheet is scanned for rows that hold both a Census code and a SOC code.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402

CENSUS = re.compile(r"^\d{4}$")
SOC = re.compile(r"^\d{2}-[\dX]{4}$")


def _cell(v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().upper()


def parse(xlsx: Path) -> pd.DataFrame:
    rows = []
    for sheet, df in pd.read_excel(xlsx, sheet_name=None, header=None, dtype=object).items():
        for r in df.itertuples(index=False):
            cells = [_cell(v) for v in r]
            census = [c.zfill(4) for c in cells if CENSUS.match(c.zfill(4)) and c.isdigit() and len(c) <= 4]
            soc = [c for c in cells if SOC.match(c)]
            if census and soc:
                rows.append({"census_occ": int(census[0]), "soc_raw": soc[0], "sheet": sheet})
    out = pd.DataFrame(rows).drop_duplicates(["census_occ", "soc_raw"])
    if out.empty:
        raise SystemExit(f"No Census/SOC code pairs found in {xlsx}")
    best = out["sheet"].value_counts().index[0]  # the code list sheet has by far the most pairs
    return out[out["sheet"] == best]


def expand(pairs: pd.DataFrame, socs: list[str]) -> pd.DataFrame:
    rows = []
    for census, raw in pairs[["census_occ", "soc_raw"]].itertuples(index=False):
        if raw in socs:
            hits = [raw]
        else:  # aggregate: 13-20XX, or a broad group written with trailing zeros such as 11-2030
            prefix = raw.split("X")[0].rstrip("0").rstrip("-")
            hits = [s for s in socs if s.startswith(prefix)] or [raw]
        rows += [{"census_occ": census, "soc": s} for s in hits]
    return pd.DataFrame(rows).drop_duplicates()


def main(argv=None) -> int:
    cfg = load_config()
    raw = resolve(cfg["paths"]["raw"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=raw / "crosswalk" / "2018-occupation-code-list-and-crosswalk.xlsx")
    a = ap.parse_args(argv)
    occ = pd.read_csv(resolve(cfg["paths"]["processed"]) / "occupations.csv", dtype={"soc": str})
    pairs = parse(a.xlsx.expanduser())
    xw = expand(pairs, sorted(occ["soc"]))
    out = raw / "crosswalk" / "census_occ_to_soc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    xw.to_csv(out, index=False)
    matched = xw["soc"].isin(occ["soc"])
    print(f"sheet '{pairs['sheet'].iloc[0]}': {len(pairs)} Census codes, {pairs['soc_raw'].str.contains('X').sum()} with X")
    print(f"wrote {out}: {len(xw)} rows, {xw['census_occ'].nunique()} Census codes, "
          f"{matched.mean():.0%} of rows match a SOC in occupations.csv")
    lost = sorted(set(pairs["census_occ"]) - set(xw.loc[matched, "census_occ"]))
    if lost:
        print(f"{len(lost)} Census codes without exposure data (dropped later): {lost[:15]}{' ...' if len(lost) > 15 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
