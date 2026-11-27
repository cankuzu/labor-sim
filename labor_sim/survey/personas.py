"""Personas for Experiment 1 (silicon sampling, Argyle et al. 2023).

from_pew()      samples real respondents from the Pew Wave 157 file, using the
                survey weights, and keeps their demographics.
synthetic()     random personas for tests and dry runs only (NOT for results).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class Persona:
    persona_id: str
    age: int
    gender: str
    education: str
    income: str
    occupation_group: str
    source: str = "pew"

    @property
    def age_group(self) -> str:
        return "18-29" if self.age < 30 else "30-49" if self.age < 50 else "50-64" if self.age < 65 else "65+"

    def backstory(self) -> str:
        return (f"I am a {self.age}-year-old {self.gender.lower()} living in the United States. "
                f"My highest level of education is {self.education.lower()}, my yearly household income is "
                f"{self.income.lower()}, and I work in {_field(self.occupation_group)}.")

    def to_dict(self) -> dict:
        return asdict(self) | {"age_group": self.age_group}


def _field(industry: str) -> str:
    return "a field not listed in the survey" if industry.strip().lower() == "other" else industry.lower()


# Pew Wave 157 variables (checked against "ATP W157 Codebook.xlsx"). Values are the
# labels pyreadstat returns with apply_value_formats=True. Pew asks no occupation
# question in this wave, so the persona's field of work is the industry item.
PEW_COLUMNS = {
    "age": "F_AGECAT",                       # 18-29 / 30-49 / 50-64 / 65+
    "gender": "F_GENDER",                    # A man / A woman / In some other way
    "education": "F_EDUCCAT2",               # 6 levels, Less than high school ... Postgraduate
    "income": "F_INC_SDT1",                  # family income, 9 brackets
    "occupation_group": "INDUSTRYCOMBO_W157",
    "weight": "WEIGHT_W157",
}
# Readme: published results exclude workers with several jobs and no primary one.
PEW_EXCLUDE = {"JOBNUM_W157": "More than one and do not consider any of them my primary job"}
AGE_RANGES = {"18-29": (18, 29), "30-49": (30, 49), "50-64": (50, 64), "65+": (65, 75)}
GENDER_WORDS = {"A man": "man", "A woman": "woman", "In some other way": "person"}


def is_missing(v: object) -> bool:
    """Pew codes 99 as 'Refused' / "Don't know/Refused/Web blank"."""
    return v is None or (isinstance(v, float) and np.isnan(v)) or "refused" in str(v).lower()


def pew_eligible(df: pd.DataFrame, columns: dict | None = None) -> pd.DataFrame:
    """Respondents usable as personas: no refused profile item, published-analysis filter applied."""
    cols = columns or PEW_COLUMNS
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Pew columns not found: {missing}. Check PEW_COLUMNS against the Wave 157 codebook.")
    keep = pd.Series(True, index=df.index)
    for var, bad in PEW_EXCLUDE.items():
        if var in df:
            keep &= df[var].astype(str) != bad
    for k, var in cols.items():
        if k != "weight":
            keep &= ~df[var].map(is_missing)
    return df[keep]


def from_pew(df: pd.DataFrame, n: int, seed: int = 0, columns: dict | None = None) -> list[Persona]:
    """Draw n respondents with probability proportional to WEIGHT_W157 and copy their profile.
    Pew only gives an age bracket, so the exact age is drawn uniformly within it."""
    cols = columns or PEW_COLUMNS
    df = pew_eligible(df, cols)
    w = df[cols["weight"]].astype(float)
    sample = df.sample(n=n, replace=True, weights=w / w.sum(), random_state=seed)
    rng = np.random.default_rng(seed)
    out = []
    for i, (_, r) in enumerate(sample.iterrows()):
        lo, hi = AGE_RANGES.get(str(r[cols["age"]]), (30, 49))
        gender = GENDER_WORDS.get(str(r[cols["gender"]]), str(r[cols["gender"]]))
        out.append(Persona(f"pew-{seed}-{i}", int(rng.integers(lo, hi + 1)), gender, str(r[cols["education"]]),
                           str(r[cols["income"]]), str(r[cols["occupation_group"]])))
    return out


def synthetic(n: int, seed: int = 0) -> list[Persona]:
    rng = np.random.default_rng(seed)
    educ = ["High school or less", "Some college", "Bachelor's degree", "Postgraduate degree"]
    inc = ["Under $50,000", "$50,000 to $99,999", "$100,000 or more"]
    occ = ["Office and administrative support", "Computer and mathematical work", "Healthcare",
           "Sales", "Education", "Production and manufacturing", "Business and financial operations"]
    return [Persona(f"syn-{seed}-{i}", int(rng.integers(18, 65)), str(rng.choice(["Woman", "Man"])),
                    str(rng.choice(educ)), str(rng.choice(inc)), str(rng.choice(occ)), source="synthetic")
            for i in range(n)]
