"""Validation targets for Experiment 2.

IPUMS CPS  -> employment and job-finding by age group x exposure quintile, per quarter
Indeed     -> job-posting trends for high- vs low-exposure sectors, before/after ChatGPT

IPUMS extract: request YEAR, MONTH, CPSIDP, AGE, EMPSTAT, OCC, WTFINL for the
basic monthly samples 2021-2025 and download as CSV (data/raw/ipums/cps.csv or .csv.gz).
CPS records Census occupation codes, not SOC; put the Census->SOC list in
data/raw/crosswalk/census_occ_to_soc.csv with columns `census_occ,soc`
(built from the Census Bureau's occupation code list). Each Census code may map
to several SOC codes; exposure is then averaged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CHATGPT_RELEASE = pd.Timestamp("2022-11-30")
EMPLOYED = {10, 12}
UNEMPLOYED = {20, 21, 22}


def _quarter(year: pd.Series, month: pd.Series) -> pd.Series:
    return year.astype(str) + "Q" + ((month.astype(int) - 1) // 3 + 1).astype(str)


def occupation_exposure_for_census(occ: pd.DataFrame, crosswalk: pd.DataFrame, source: str) -> pd.DataFrame:
    """Average exposure percentile per Census occupation code."""
    col = f"pctl_{source}"
    m = crosswalk.merge(occ[["soc", col]], on="soc", how="inner")
    out = m.groupby("census_occ", as_index=False)[col].mean()
    out["quintile"] = pd.qcut(out[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return out


def ipums_targets(cps: pd.DataFrame, census_exposure: pd.DataFrame, young_max: int = 25) -> pd.DataFrame:
    """Quarterly employment rate and job-finding rate by age group x exposure quintile.

    Job finding = share of people unemployed in month t who are employed in t+1
    (linked with CPSIDP), assigned to the quintile of the occupation they report.
    """
    cps = cps.rename(columns=str.upper)
    df = cps[(cps["AGE"] >= 22) & (cps["AGE"] <= 64)].copy()
    df["age_group"] = np.where(df["AGE"] <= young_max, f"22-{young_max}", f"{young_max + 1}+")
    df = df.merge(census_exposure.rename(columns={"census_occ": "OCC"}), on="OCC", how="inner")
    df["employed"] = df["EMPSTAT"].isin(EMPLOYED)
    df["unemployed"] = df["EMPSTAT"].isin(UNEMPLOYED)
    df["t"] = df["YEAR"] * 12 + df["MONTH"]
    df["quarter"] = _quarter(df["YEAR"], df["MONTH"])

    nxt = df[["CPSIDP", "t", "employed"]].rename(columns={"employed": "employed_next"})
    nxt["t"] -= 1
    flows = df[df["unemployed"]].merge(nxt, on=["CPSIDP", "t"], how="inner")

    w = "WTFINL" if "WTFINL" in df else None
    def wmean(g: pd.DataFrame, col: str) -> float:
        return float(np.average(g[col], weights=g[w])) if w else float(g[col].mean())

    stock = df.groupby(["quarter", "age_group", "quintile"]).apply(
        lambda g: pd.Series({"employment_rate": wmean(g, "employed"), "n": len(g)}), include_groups=False)
    find = flows.groupby(["quarter", "age_group", "quintile"]).apply(
        lambda g: pd.Series({"job_finding_rate": wmean(g, "employed_next"), "n_flows": len(g)}), include_groups=False)
    return stock.join(find, how="left").reset_index()


def indeed_targets(postings: pd.DataFrame, sector_map: pd.DataFrame, occ: pd.DataFrame, source: str) -> pd.DataFrame:
    """Monthly postings index for each sector, with sector exposure and pre/post-ChatGPT change."""
    col = f"pctl_{source}"
    sm = sector_map.copy()
    rows = []
    for _, r in sm.iterrows():
        prefixes = [p.strip() for p in str(r["soc_prefixes"]).split(";") if p.strip()]
        hit = occ[occ["soc"].astype(str).str.startswith(tuple(prefixes))]
        emp_col = sorted(c for c in occ.columns if c.startswith("emp_"))[-1]
        wts = hit[emp_col].fillna(0)
        exp = float(np.average(hit[col], weights=wts)) if wts.sum() > 0 and hit[col].notna().all() else float(hit[col].mean())
        rows.append({"display_name": r["sector"], "sector_exposure": exp, "n_occ": len(hit)})
    sector_exp = pd.DataFrame(rows)

    p = postings[postings["variable"] == "total postings"].copy()
    p["date"] = pd.to_datetime(p["date"])
    p["month"] = p["date"].dt.to_period("M").dt.to_timestamp()
    monthly = p.groupby(["display_name", "month"], as_index=False)["indeed_job_postings_index"].mean()
    pre = monthly[(monthly["month"] >= "2022-05-01") & (monthly["month"] < "2022-12-01")].groupby("display_name")["indeed_job_postings_index"].mean()
    monthly = monthly.merge(pre.rename("pre_chatgpt_level").reset_index(), on="display_name")
    monthly["change_vs_pre"] = np.log(monthly["indeed_job_postings_index"] / monthly["pre_chatgpt_level"])
    monthly = monthly.merge(sector_exp, on="display_name", how="left")
    monthly["exposure_group"] = pd.qcut(monthly["sector_exposure"].rank(method="dense"), 3, labels=["low", "mid", "high"])
    return monthly


def load_cps(raw: Path) -> pd.DataFrame | None:
    for name in ["cps.csv", "cps.csv.gz", "cps.dta"]:
        p = raw / "ipums" / name
        if p.exists():
            return pd.read_stata(p) if p.suffix == ".dta" else pd.read_csv(p)
    return None
