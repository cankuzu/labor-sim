"""Build the merged occupation table: one row per 6-digit SOC code.

Columns (when the source is available):
    soc, title, major_group
    emp_<year>, wage_<year>           BLS OEWS (May of each year)
    emp_change_2022_2025              target for the Random Forest baseline
    n_tasks                           O*NET task count
    exposure_<source>                 raw score from each source
    pctl_<source>                     percentile rank 0-1 (comparable across sources)
    quintile_<source>                 1 (lowest) .. 5 (highest)
    automation_share                  AEI share of automation vs augmentation

Sources that are missing are skipped with a message, so the table can be
built step by step while data are still being downloaded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPOSURE_SOURCES = ["anthropic", "eloundou", "microsoft", "aioe"]


def soc6(code: str) -> str:
    """'15-1252.00' or '151252' -> '15-1252'."""
    s = str(code).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 6:
        return ""
    return f"{digits[:2]}-{digits[2:6]}"


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "").str.strip(), errors="coerce")


# --------------------------------------------------------------------------- sources

def load_eloundou(raw: Path) -> pd.DataFrame | None:
    p = raw / "eloundou" / "occ_level.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["soc"] = df["O*NET-SOC Code"].map(soc6)
    # beta = E1 + 0.5*E2, the headline measure in Eloundou et al. (2024)
    out = df.groupby("soc").agg(title_eloundou=("Title", "first"),
                                exposure_eloundou=("dv_rating_beta", "mean"),
                                exposure_eloundou_human=("human_rating_beta", "mean"))
    return out.reset_index()


def load_microsoft(raw: Path) -> pd.DataFrame | None:
    p = raw / "microsoft" / "ai_applicability_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["soc"] = df["SOC Code"].map(soc6)
    return df.rename(columns={"title": "title_microsoft",
                              "ai_applicability_score": "exposure_microsoft"})[["soc", "title_microsoft", "exposure_microsoft"]]


def load_soc_crosswalk(raw: Path) -> pd.DataFrame | None:
    """BLS SOC 2010 -> 2018 crosswalk (data/raw/crosswalk/soc_2010_to_2018_crosswalk.xlsx)."""
    p = raw / "crosswalk" / "soc_2010_to_2018_crosswalk.xlsx"
    if not p.exists():
        return None
    df = pd.read_excel(p, header=None)
    # the header row is the first row that mentions both SOC years
    hdr = df.index[df.astype(str).apply(lambda r: r.str.contains("2010 SOC Code").any(), axis=1)][0]
    df.columns = df.iloc[hdr].astype(str).str.strip()
    df = df.iloc[hdr + 1:]
    c10 = next(c for c in df.columns if "2010 SOC Code" in c)
    c18 = next(c for c in df.columns if "2018 SOC Code" in c)
    return pd.DataFrame({"soc2010": df[c10].map(soc6), "soc": df[c18].map(soc6)}).dropna()


def load_aioe(raw: Path) -> pd.DataFrame | None:
    """AIOE is coded in SOC 2010; mapped to SOC 2018 when the BLS crosswalk is present."""
    p = raw / "aioe" / "AIOE_DataAppendix.xlsx"
    if not p.exists():
        return None
    df = pd.read_excel(p, sheet_name="Appendix A")
    df = df.rename(columns={"Occupation Title": "title_aioe", "AIOE": "exposure_aioe"})
    df["soc"] = df["SOC Code"].map(soc6)
    xw = load_soc_crosswalk(raw)
    if xw is not None:
        m = xw.merge(df[["soc", "exposure_aioe"]].rename(columns={"soc": "soc2010"}), on="soc2010")
        # several 2010 codes can merge into one 2018 code: average them
        df = m.groupby("soc", as_index=False)["exposure_aioe"].mean()
        df["title_aioe"] = None
    else:
        print("    (AIOE kept in SOC 2010; add the BLS crosswalk to map it to SOC 2018)")
    return df[["soc", "title_aioe", "exposure_aioe"]]


def load_anthropic(raw: Path) -> pd.DataFrame | None:
    """Usage share and automation share per SOC from the Anthropic Economic Index.

    Uses the first public release layout (release_2025_02_10):
      onet_task_mappings.csv                 task_name, pct
      automation_vs_augmentation_by_task.csv task_name + interaction-pattern shares
      onet_task_statements.csv               O*NET-SOC Code, Task
    Later releases add more files; check data/raw/anthropic/ and adapt the
    file names below if the layout differs.
    """
    base = raw / "anthropic"
    if not base.exists():
        return None

    def find(name: str) -> Path | None:
        hits = sorted(base.rglob(name))
        return hits[0] if hits else None

    tasks_p, auto_p, stmt_p = (find("onet_task_mappings.csv"),
                               find("automation_vs_augmentation_by_task.csv"),
                               find("onet_task_statements.csv"))
    if not (tasks_p and stmt_p):
        print("  ! Anthropic files not in the expected layout; skipping (see load_anthropic docstring)")
        return None

    tasks = pd.read_csv(tasks_p)
    tasks["task_key"] = tasks["task_name"].str.lower().str.strip()
    stmt = pd.read_csv(stmt_p)
    stmt["task_key"] = stmt["Task"].str.lower().str.strip()
    stmt["soc"] = stmt["O*NET-SOC Code"].map(soc6)
    # a task can appear in several occupations: split its usage share evenly
    stmt["n_occ"] = stmt.groupby("task_key")["soc"].transform("nunique")
    m = stmt.merge(tasks[["task_key", "pct"]], on="task_key", how="inner")
    m["pct_share"] = m["pct"] / m["n_occ"]
    out = m.groupby("soc")["pct_share"].sum().rename("exposure_anthropic").reset_index()

    if auto_p:
        a = pd.read_csv(auto_p)
        a["task_key"] = a["task_name"].str.lower().str.strip()
        auto_cols = [c for c in ["directive", "feedback_loop"] if c in a]
        aug_cols = [c for c in ["task_iteration", "learning", "validation"] if c in a]
        if auto_cols and aug_cols:
            a["auto"] = a[auto_cols].sum(axis=1)
            a["aug"] = a[aug_cols].sum(axis=1)
            ma = stmt.merge(a[["task_key", "auto", "aug"]], on="task_key")
            g = ma.groupby("soc")[["auto", "aug"]].sum()
            share = (g["auto"] / (g["auto"] + g["aug"])).rename("automation_share")
            out = out.merge(share.reset_index(), on="soc", how="left")
    return out


def load_oews(raw: Path) -> pd.DataFrame | None:
    """Employment and median wage per detailed SOC for each available May file."""
    frames = []
    year_files: dict[int, Path] = {}
    oews_dir = raw / "oews"
    if oews_dir.exists():
        for ydir in sorted(oews_dir.iterdir()):
            hits = sorted(ydir.rglob("national_M*_dl.xlsx")) or sorted(ydir.rglob("*.xlsx"))
            if hits and ydir.name.isdigit():
                year_files[int(ydir.name)] = hits[0]
    dev = raw / "eloundou" / "national_May2021_dl.csv"
    if dev.exists():
        year_files.setdefault(2021, dev)
    if not year_files:
        return None
    for year, p in sorted(year_files.items()):
        df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_excel(p)
        df.columns = [c.upper() for c in df.columns]
        df = df[df["O_GROUP"].astype(str).str.lower() == "detailed"]
        if "AREA" in df:
            df = df[df["AREA"].astype(str) == "99"]
        if "NAICS" in df:
            df = df[df["NAICS"].astype(str).isin(["0", "000000"])]
        f = pd.DataFrame({"soc": df["OCC_CODE"].map(soc6),
                          "title_oews": df["OCC_TITLE"],
                          f"emp_{year}": _num(df["TOT_EMP"]),
                          f"wage_{year}": _num(df["A_MEDIAN"])})
        frames.append(f.drop_duplicates("soc").set_index("soc"))
    out = pd.concat(frames, axis=1)
    out["title_oews"] = out.filter(like="title_oews").bfill(axis=1).iloc[:, 0]
    out = out.loc[:, ~out.columns.duplicated()]
    return out.reset_index()


def load_onet_tasks(raw: Path) -> pd.DataFrame | None:
    p = next(iter(sorted((raw / "onet").rglob("Task Statements.txt"))), None) if (raw / "onet").exists() else None
    if p is not None:
        df = pd.read_csv(p, sep="\t")
    else:
        p = raw / "eloundou" / "full_onet_data.tsv"
        if not p.exists():
            return None
        df = pd.read_csv(p, sep="\t")
    df["soc"] = df["O*NET-SOC Code"].map(soc6)
    return df.groupby("soc").size().rename("n_tasks").reset_index()


# --------------------------------------------------------------------------- build

def build_occupations(raw: Path) -> tuple[pd.DataFrame, dict]:
    report: dict[str, object] = {}
    loaders = {"oews": load_oews, "onet": load_onet_tasks, "eloundou": load_eloundou,
               "microsoft": load_microsoft, "aioe": load_aioe, "anthropic": load_anthropic}
    tables = {}
    for name, fn in loaders.items():
        t = fn(raw)
        report[f"{name}_rows"] = 0 if t is None else len(t)
        if t is None:
            print(f"  - {name}: not found, skipped")
        else:
            tables[name] = t
            print(f"  + {name}: {len(t)} occupations")

    if "oews" not in tables:
        raise FileNotFoundError("OEWS data are required (data/raw/oews/<year>/ or the 2021 dev file).")
    occ = tables["oews"]
    for name in ["onet", "eloundou", "microsoft", "aioe", "anthropic"]:
        if name in tables:
            before = set(tables[name]["soc"])
            occ = occ.merge(tables[name], on="soc", how="left")
            report[f"{name}_unmatched_soc"] = sorted(before - set(occ["soc"]))[:50]

    occ["title"] = occ["title_oews"]
    occ = occ.drop(columns=[c for c in occ.columns if c.startswith("title_")])
    occ["major_group"] = occ["soc"].str[:2]

    for src in EXPOSURE_SOURCES:
        col = f"exposure_{src}"
        if col in occ:
            occ[f"pctl_{src}"] = occ[col].rank(pct=True)
            occ[f"quintile_{src}"] = pd.qcut(occ[col].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype("Int64")

    emp_years = sorted(int(c[4:]) for c in occ.columns if c.startswith("emp_") and c[4:].isdigit())
    report["emp_years"] = emp_years
    if 2022 in emp_years and 2025 in emp_years:
        occ["emp_change_2022_2025"] = np.log(occ["emp_2025"]) - np.log(occ["emp_2022"])
    report["n_occupations"] = len(occ)
    report["coverage"] = {c: round(float(occ[c].notna().mean()), 3) for c in occ.columns if c.startswith("exposure_")}
    return occ, report
