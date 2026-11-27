"""Experiment 2: quarterly agent-based labor market, 2022Q4-2025Q4.

Each quarter:
  1. AI capability rises in every employer cell (logistic path scaled by the cell's exposure).
  2. Workers age; those reaching 65 retire, which opens a vacancy.
  3. New 22-year-old entrants join as job seekers in the field they trained for.
  4. Baseline quits open vacancies; quitters become job seekers.
  5. Employer cells decide on open vacancies (hire / delay / augment with AI) and layoffs.
  6. Workers decide (unemployed: search / upskill / switch occupation; employed: adopt AI / upskill / stay).
  7. Rule-based matching fills "hire" slots from searchers in the same cell.
  8. Metrics are recorded by age group x exposure quintile, plus employment per cell.

Employer cells = SOC major group x exposure quintile, so every cell has enough
workers to make one decision per quarter.
"""
from __future__ import annotations

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from labor_sim.sim.agents import Employer, Worker
from labor_sim.sim.policies import RulePolicy

SOC_MAJOR = {
    "11": "Management", "13": "Business and financial operations", "15": "Computer and mathematical",
    "17": "Architecture and engineering", "19": "Life, physical and social science",
    "21": "Community and social service", "23": "Legal", "25": "Educational instruction and library",
    "27": "Arts, design, entertainment, sports and media", "29": "Healthcare practitioners and technical",
    "31": "Healthcare support", "33": "Protective service", "35": "Food preparation and serving",
    "37": "Building and grounds cleaning and maintenance", "39": "Personal care and service",
    "41": "Sales and related", "43": "Office and administrative support", "45": "Farming, fishing and forestry",
    "47": "Construction and extraction", "49": "Installation, maintenance and repair",
    "51": "Production", "53": "Transportation and material moving",
}
# Rough age structure of the US workforce aged 22-64 (bands and shares; adjust to CPS if needed)
AGE_BANDS = [((22, 25), 0.09), ((26, 34), 0.21), ((35, 44), 0.22), ((45, 54), 0.21), ((55, 64), 0.27)]


def quarters(start: str, end: str) -> list[str]:
    y, q = int(start[:4]), int(start[-1])
    ye, qe = int(end[:4]), int(end[-1])
    out = []
    while (y, q) <= (ye, qe):
        out.append(f"{y}Q{q}")
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
    return out


@dataclass
class SimConfig:
    n_workers: int = 500
    start_quarter: str = "2022Q4"
    end_quarter: str = "2025Q4"
    exposure_source: str = "eloundou"
    seed: int = 1
    quarterly_separation_rate: float = 0.035
    quarterly_entry_rate: float = 0.012
    quarterly_job_creation_rate: float = 0.010
    young_age_max: int = 25
    ai_capability_growth: float = 0.25
    initial_ai_adoption: float = 0.265
    time_saving_per_adopter: float = 0.028
    employer_encouragement_boost: float = 0.36
    novice_productivity_gain: float = 0.15
    default_automation_share: float = 0.43
    bass_innovation: float = 0.04
    bass_imitation: float = 0.40
    llm_worker_decisions_every: int = 1
    initial_unemployment: float = 0.04

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SimResult:
    run_id: str
    config: dict
    policy: str
    rounds: list[dict] = field(default_factory=list)
    status: str = "running"
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def metrics_frame(self) -> pd.DataFrame:
        return pd.DataFrame([m | {"quarter": r["quarter"]} for r in self.rounds for m in r["groups"]])

    def cells_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c | {"quarter": r["quarter"]} for r in self.rounds for c in r["cells"]])


class Simulation:
    def __init__(self, occupations: pd.DataFrame, cfg: SimConfig, policy=None, run_id: str = "run",
                 out_dir: Path | None = None):
        self.cfg, self.run_id, self.out_dir = cfg, run_id, out_dir
        self.rng = random.Random(cfg.seed)
        self.policy = policy or RulePolicy(asdict(cfg), self.rng)
        self.quarters = quarters(cfg.start_quarter, cfg.end_quarter)
        self.occ = self._prepare(occupations)
        self.employers: dict[str, Employer] = {}
        self.workers: list[Worker] = []
        self._next_id = 0
        self._init_population()
        self.result = SimResult(run_id, asdict(cfg), getattr(self.policy, "name", "rule"))

    # ------------------------------------------------------------------ setup
    def _prepare(self, occ: pd.DataFrame) -> pd.DataFrame:
        src = self.cfg.exposure_source
        pcol = f"pctl_{src}"
        if pcol not in occ:
            raise KeyError(f"exposure source '{src}' not in occupation table (missing {pcol})")
        emp_col = sorted(c for c in occ.columns if c.startswith("emp_") and c[4:].isdigit())[0]
        o = occ[["soc", "title", pcol, emp_col] + (["automation_share"] if "automation_share" in occ else [])].copy()
        o = o.dropna(subset=[pcol, emp_col])
        o["soc"] = o["soc"].astype(str)
        o["major_group"] = o["soc"].str[:2]
        o = o[o["major_group"].isin(SOC_MAJOR)]
        o["quintile"] = pd.qcut(o[pcol].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
        o["cell"] = o["major_group"] + "-q" + o["quintile"].astype(str)
        o = o.rename(columns={pcol: "exposure", emp_col: "employment"})
        if "automation_share" not in o:
            o["automation_share"] = np.nan
        o["automation_share"] = o["automation_share"].fillna(self.cfg.default_automation_share)
        return o.reset_index(drop=True)

    def _new_worker(self, row: pd.Series, age: float, employed: bool) -> Worker:
        exp = max(0.0, age - 22 - self.rng.uniform(0, 3))
        w = Worker(self._next_id, age, row["soc"], row["title"], row["cell"], exp, employed,
                   ai_adopter=self.rng.random() < self.cfg.initial_ai_adoption * (1.2 if age < 30 else 1.0) * row["exposure"] * 2)
        self._next_id += 1
        return w

    def _sample_occ(self, k: int) -> pd.DataFrame:
        return self.occ.sample(n=k, replace=True, weights="employment", random_state=self.rng.randrange(10**9))

    def _init_population(self) -> None:
        for cell, g in self.occ.groupby("cell"):
            mg = cell.split("-")[0]
            w = g["employment"]
            self.employers[cell] = Employer(cell, mg, SOC_MAJOR[mg], int(g["quintile"].iloc[0]),
                                            float(np.average(g["exposure"], weights=w)),
                                            float(np.average(g["automation_share"], weights=w)))
        bands = [b for b, _ in AGE_BANDS]
        shares = [s for _, s in AGE_BANDS]
        for _, row in self._sample_occ(self.cfg.n_workers).iterrows():
            lo, hi = self.rng.choices(bands, weights=shares)[0]
            age = self.rng.uniform(lo, hi + 1)
            self.workers.append(self._new_worker(row, age, self.rng.random() > self.cfg.initial_unemployment))

    # ------------------------------------------------------------------ dynamics
    def capability(self, e: Employer, t: int) -> float:
        """AI capability in a cell: logistic in time, scaled by the cell's exposure (0-1)."""
        mid = len(self.quarters) / 2
        s = 1 / (1 + math.exp(-self.cfg.ai_capability_growth * (t - mid)))
        s0 = 1 / (1 + math.exp(self.cfg.ai_capability_growth * mid))
        return e.exposure * (s - s0) / (1 - s0)

    def _switch_target(self, w: Worker) -> pd.Series:
        """Retrain toward a less exposed occupation in the same major group (occupation-proximity path)."""
        cur = self.employers[w.cell]
        same = self.occ[(self.occ["major_group"] == cur.major_group) & (self.occ["quintile"] < cur.quintile)]
        pool = same if len(same) else self.occ[self.occ["quintile"] < cur.quintile]
        pool = pool if len(pool) else self.occ
        return pool.sample(1, weights="employment", random_state=self.rng.randrange(10**9)).iloc[0]

    def step(self, t: int) -> dict:
        q = self.quarters[t]
        c = self.cfg
        young = lambda w: w.age <= c.young_age_max  # noqa: E731
        by_cell: dict[str, list[Worker]] = defaultdict(list)
        for w in self.workers:
            by_cell[w.cell].append(w)
        caps = {cell: self.capability(e, t) for cell, e in self.employers.items()}
        start_state = {w.wid: (w.employed, w.cell, w.age_group(c.young_age_max)) for w in self.workers}
        vac: Counter = Counter({cell: e.open_vacancies for cell, e in self.employers.items()})
        flows: Counter = Counter()

        # ageing and retirement
        if q.endswith("Q1"):
            for w in self.workers:
                w.age += 1
                w.experience += 1 if w.employed else 0
        retired = [w for w in self.workers if w.age >= 65]
        for w in retired:
            if w.employed:
                vac[w.cell] += 1
            self.workers.remove(w)

        # entrants: new graduates looking for their first job
        n_entry = np.random.default_rng(self.rng.randrange(10**9)).poisson(c.quarterly_entry_rate * len(self.workers))
        for _, row in self._sample_occ(int(n_entry)).iterrows():
            nw = self._new_worker(row, 22 + self.rng.random(), employed=False)
            nw.experience = 0.0
            nw.remember(f"{q}: graduated and started looking for work")
            self.workers.append(nw)
            flows[("entry", nw.cell, nw.age_group(c.young_age_max))] += 1

        # quits
        for w in self.workers:
            if w.employed and self.rng.random() < c.quarterly_separation_rate:
                w.employed = False
                w.quarters_unemployed = 0
                vac[w.cell] += 1
                w.remember(f"{q}: left previous job")
                flows[("quit", w.cell, w.age_group(c.young_age_max))] += 1

        # baseline job creation: keeps the economy absorbing new entrants when there is no AI effect
        rng_np = np.random.default_rng(self.rng.randrange(10**9))
        for cell, ws in by_cell.items():
            staff = sum(w.employed for w in ws)
            vac[cell] += int(rng_np.poisson(c.quarterly_job_creation_rate * staff))

        # productivity: adopters save time, so cells need slightly fewer replacements
        for cell, ws in by_cell.items():
            emp = [w for w in ws if w.employed]
            if emp and vac[cell]:
                saved = c.time_saving_per_adopter * sum(w.ai_adopter for w in emp) / len(emp) * caps[cell]
                vac[cell] = max(0, round(vac[cell] * (1 - saved)))

        # employer decisions
        adopted_share = sum(e.adopted_ai for e in self.employers.values()) / max(1, len(self.employers))
        decisions = {}
        for cell, e in self.employers.items():
            ws = [w for w in self.workers if w.cell == cell]
            staff = sum(w.employed for w in ws)
            seekers = [w for w in ws if not w.employed]
            if not ws or (vac[cell] == 0 and staff == 0):
                continue
            d = self.policy.employer(e, caps[cell], int(vac[cell]), staff,
                                     {"young": sum(young(w) for w in seekers),
                                      "experienced": sum(not young(w) for w in seekers)},
                                     adopted_share, q)
            decisions[cell] = d
            e.open_vacancies = d.delay
            if d.augment:
                e.remember(f"{q}: covered {d.augment} position(s) with AI")
            if d.layoffs:
                for w in self.rng.sample([w for w in ws if w.employed], min(d.layoffs, staff)):
                    w.employed, w.quarters_unemployed = False, 0
                    w.remember(f"{q}: laid off")
                    flows[("layoff", w.cell, w.age_group(c.young_age_max))] += 1

        # worker decisions
        ask_llm = t % max(1, c.llm_worker_decisions_every) == 0
        searching: list[Worker] = []
        current = list(self.workers)
        batch = None
        if ask_llm and hasattr(self.policy, "workers_batch"):
            # LLM workers decide on this quarter's state; requests can run concurrently
            batch = self.policy.workers_batch([(w, caps[w.cell], self.employers[w.cell].adopted_ai, q)
                                               for w in current])
        for i, w in enumerate(current):
            e = self.employers[w.cell]
            if batch is not None:
                wd = batch[i]
            elif ask_llm or not hasattr(self.policy, "client"):
                wd = self.policy.worker(w, caps[w.cell], e.adopted_ai, q)
            else:
                wd = RulePolicy.worker(self.policy, w, caps[w.cell], e.adopted_ai, q)
            if wd.action == "adopt_ai":
                w.ai_adopter = True
            elif wd.action == "upskill":
                w.upskilled += 1
                w.remember(f"{q}: took a training course")
            elif wd.action == "switch_occupation":
                row = self._switch_target(w)
                w.soc, w.title, w.cell, w.experience = row["soc"], row["title"], row["cell"], 0.0
                w.remember(f"{q}: retrained as {row['title'].lower()}")
                flows[("switch", w.cell, w.age_group(c.young_age_max))] += 1
                searching.append(w)
            elif wd.action == "search":
                searching.append(w)
            if not w.employed and wd.action == "upskill":
                pass  # training this quarter, not searching

        # matching
        hires_log = []
        for cell, d in decisions.items():
            pool = [w for w in searching if w.cell == cell and not w.employed]
            e = self.employers[cell]
            for _ in range(d.hire):
                if not pool:
                    e.open_vacancies += 1  # unfilled: stays open
                    continue
                wts = [self.policy.candidate_weight(w, e, caps[cell], d.prefer) for w in pool]
                w = self.rng.choices(pool, weights=wts)[0]
                pool.remove(w)
                w.employed, w.quarters_unemployed = True, 0
                w.remember(f"{q}: hired")
                flows[("hire", cell, w.age_group(c.young_age_max))] += 1
                hires_log.append(w.wid)
        for w in self.workers:
            if not w.employed:
                w.quarters_unemployed += 1

        return self._record(t, q, caps, decisions, flows, start_state, searching)

    # ------------------------------------------------------------------ metrics
    def _record(self, t, q, caps, decisions, flows, start_state, searching) -> dict:
        c = self.cfg
        groups = []
        searching_ids = {w.wid for w in searching}
        for (ag, quint), ws in self._group_workers().items():
            emp = sum(w.employed for w in ws)
            seekers_start = [w for w in ws if w.wid in searching_ids and not start_state.get(w.wid, (True,))[0]]
            found = sum(w.employed for w in seekers_start)
            hires = sum(v for (kind, cell, a), v in flows.items()
                        if kind == "hire" and a == ag and self.employers[cell].quintile == quint)
            seps = sum(v for (kind, cell, a), v in flows.items()
                       if kind in ("quit", "layoff") and a == ag and self.employers[cell].quintile == quint)
            layoffs = sum(v for (kind, cell, a), v in flows.items()
                          if kind == "layoff" and a == ag and self.employers[cell].quintile == quint)
            groups.append({"age_group": ag, "exposure_quintile": quint, "labor_force": len(ws), "employed": emp,
                           "employment_rate": emp / len(ws) if ws else None, "hires": hires,
                           "separations": seps, "layoffs": layoffs,
                           "job_finding_rate": found / len(seekers_start) if seekers_start else None,
                           "ai_adoption_rate": sum(w.ai_adopter for w in ws if w.employed) / emp if emp else None})
        cells = []
        for cell, e in self.employers.items():
            ws = [w for w in self.workers if w.cell == cell]
            d = decisions.get(cell)
            cells.append({"cell": cell, "major_group": e.major_group, "quintile": e.quintile,
                          "capability": round(caps[cell], 4), "employed": sum(w.employed for w in ws),
                          "adopted_ai": e.adopted_ai,
                          **({"hire": d.hire, "delay": d.delay, "augment": d.augment, "layoffs": d.layoffs,
                              "prefer": d.prefer, "reason": d.reason, "valid": d.valid} if d else {})})
        rnd = {"round": t, "quarter": q, "n_workers": len(self.workers),
               "employment_rate": sum(w.employed for w in self.workers) / len(self.workers),
               "groups": groups, "cells": cells}
        self.result.rounds.append(rnd)
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / f"round_{t:02d}.json").write_text(json.dumps(rnd, default=_json_default))
        return rnd

    def _group_workers(self) -> dict:
        g: dict = defaultdict(list)
        for w in self.workers:
            g[(w.age_group(self.cfg.young_age_max), self.employers[w.cell].quintile)].append(w)
        return dict(sorted(g.items()))

    # ------------------------------------------------------------------ run
    def run(self, on_round: Callable[[dict], None] | None = None) -> SimResult:
        try:
            for t in range(len(self.quarters)):
                rnd = self.step(t)
                if on_round:
                    on_round(rnd)
            self.result.status = "finished"
        except Exception as exc:  # keep partial results
            self.result.status, self.result.error = "failed", repr(exc)
            raise
        finally:
            self.result.finished = time.time()
            if self.out_dir:
                (self.out_dir / "summary.json").write_text(json.dumps(
                    {k: v for k, v in asdict(self.result).items() if k != "rounds"} | {"n_rounds": len(self.result.rounds)},
                    default=_json_default, indent=2))
        return self.result


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
