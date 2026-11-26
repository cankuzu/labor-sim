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
                f"{self.income.lower()}, and I work in {self.occupation_group.lower()}.")

    def to_dict(self) -> dict:
        return asdict(self) | {"age_group": self.age_group}


# Map from Pew codebook variable names to persona fields. Fill in from the
# Wave 157 codebook; values are the labels pyreadstat returns.
# The names below follow Pew's usual ATP naming (F_ = profile variables,
# WEIGHT_W<wave>). They are best guesses: confirm each one in the Wave 157
# codebook. The occupation/industry variable differs between surveys.
PEW_COLUMNS = {
    "age": "F_AGECAT",            # categories 18-29 / 30-49 / 50-64 / 65+, mapped to midpoints below
    "gender": "F_GENDER",
    "education": "F_EDUCCAT",
    "income": "F_INC_TIER2",
    "occupation_group": "TODO_OCCUPATION",   # find the job/industry item in the codebook
    "weight": "WEIGHT_W157",
}
AGE_MIDPOINTS = {"18-29": 24, "30-49": 40, "50-64": 57, "65+": 70,
                 "Ages 18-29": 24, "30-49 years": 40, "50-64 years": 57, "65+ years": 70}


def from_pew(df: pd.DataFrame, n: int, seed: int = 0, columns: dict | None = None) -> list[Persona]:
    cols = columns or PEW_COLUMNS
    missing = [v for v in cols.values() if v not in df.columns]
    if missing:
        raise KeyError(f"Pew columns not found: {missing}. Set PEW_COLUMNS from the Wave 157 codebook.")
    w = df[cols["weight"]].astype(float)
    sample = df.sample(n=n, replace=True, weights=w / w.sum(), random_state=seed)
    out = []
    for i, (_, r) in enumerate(sample.iterrows()):
        age = r[cols["age"]]
        age = AGE_MIDPOINTS.get(str(age), age)
        out.append(Persona(f"pew-{seed}-{i}", int(age), str(r[cols["gender"]]), str(r[cols["education"]]),
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
