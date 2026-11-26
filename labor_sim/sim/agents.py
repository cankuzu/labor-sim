"""Agents of the labor market simulation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Worker:
    wid: int
    age: float
    soc: str
    title: str
    cell: str                    # employer cell = SOC major group x exposure quintile
    experience: float            # years in current field
    employed: bool = True
    ai_adopter: bool = False
    quarters_unemployed: int = 0
    upskilled: int = 0
    memory: list[str] = field(default_factory=list)

    def age_group(self, young_max: int) -> str:
        return f"22-{young_max}" if self.age <= young_max else f"{young_max + 1}+"

    def remember(self, event: str, keep: int = 4) -> None:
        self.memory = (self.memory + [event])[-keep:]


@dataclass
class Employer:
    cell: str
    major_group: str
    major_title: str
    quintile: int
    exposure: float              # percentile 0-1 of the cell's occupations
    automation_share: float      # share of AI use that automates rather than augments
    adopted_ai: bool = False
    open_vacancies: int = 0      # carried over (delayed) vacancies
    memory: list[str] = field(default_factory=list)

    def remember(self, event: str, keep: int = 4) -> None:
        self.memory = (self.memory + [event])[-keep:]
