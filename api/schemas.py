"""Request and response models (they also generate the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Exposure = Literal["anthropic", "eloundou", "microsoft", "aioe"]


class SurveyRequest(BaseModel):
    model: str = Field("llama-instruct", description="model key from config/default.yaml")
    n_personas: int = Field(50, ge=1, le=5000)
    seed: int = 1
    synthetic_personas: bool = Field(False, description="use synthetic personas (dry runs only)")


class SimulationRequest(BaseModel):
    policy: Literal["rule", "llm"] = "rule"
    model: str = Field("llama-instruct", description="model key, used when policy='llm'")
    exposure_source: Exposure = "eloundou"
    n_workers: int = Field(500, ge=10, le=20000)
    seed: int = 1
    start_quarter: str = "2022Q4"
    end_quarter: str = "2025Q4"


class RunStatus(BaseModel):
    id: str
    kind: Literal["survey", "simulation"]
    status: Literal["queued", "running", "finished", "failed"]
    config: dict
    rounds_done: int = 0
    rounds_total: int | None = None
    error: str | None = None
    created: float
    finished: float | None = None


class GroupMetric(BaseModel):
    age_group: str
    exposure_quintile: int
    labor_force: int
    employed: int
    employment_rate: float | None
    hires: int
    separations: int
    layoffs: int
    job_finding_rate: float | None
    ai_adoption_rate: float | None


class Round(BaseModel):
    round: int
    quarter: str
    n_workers: int
    employment_rate: float
    groups: list[GroupMetric]
    cells: list[dict]


class Metrics(BaseModel):
    id: str
    quarters: list[str]
    by_group: list[dict]
    young_exposed_gap: dict | None = Field(None, description="change in employment rate, 22-25 vs older, "
                                                             "high (Q4-5) minus low (Q1-2) exposure, first to last quarter")
