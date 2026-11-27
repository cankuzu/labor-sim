"""Decision policies.

RulePolicy  the LLM-free baseline: firms adopt AI by Bass diffusion and workers
            retrain toward less exposed occupations (Upreti & Sridhar 2024).
LLMPolicy   the same decisions made by an LLM, one call per employer cell and
            one call per worker each quarter, answered as JSON.

Both return the same structures so the engine does not care which one runs.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from labor_sim.llm.client import LLMClient, parse_json
from labor_sim.sim.agents import Employer, Worker

EMPLOYER_ACTIONS = ("hire", "delay", "augment")
UNEMPLOYED_ACTIONS = ("search", "upskill", "switch_occupation")
EMPLOYED_ACTIONS = ("adopt_ai", "upskill", "stay")


@dataclass
class EmployerDecision:
    hire: int
    delay: int
    augment: int
    layoffs: int
    prefer: str = "no_preference"     # experienced | junior | no_preference
    reason: str = ""
    raw: str = ""
    valid: bool = True


@dataclass
class WorkerDecision:
    action: str
    reason: str = ""
    raw: str = ""
    valid: bool = True


def _level(x: float) -> str:
    return "very low" if x < 0.2 else "low" if x < 0.4 else "moderate" if x < 0.6 else "high" if x < 0.8 else "very high"


# --------------------------------------------------------------------------- rule-based

class RulePolicy:
    name = "rule"

    def __init__(self, cfg: dict, rng: random.Random):
        self.c, self.rng = cfg, rng

    def employer(self, e: Employer, cap: float, vacancies: int, staff: int, applicants: dict,
                 adopted_share: float, quarter: str) -> EmployerDecision:
        if not e.adopted_ai:
            # Bass diffusion; only cells where AI can do a meaningful share of tasks adopt at full speed
            p = (self.c["bass_innovation"] + self.c["bass_imitation"] * adopted_share) * min(1.0, 2 * cap)
            e.adopted_ai = self.rng.random() < p
        hire = delay = augment = 0
        for _ in range(vacancies):
            if e.adopted_ai and self.rng.random() < cap * e.automation_share * 0.5:
                augment += 1
            elif self.rng.random() < 0.1 * cap:
                delay += 1
            else:
                hire += 1
        layoffs = sum(self.rng.random() < cap * e.automation_share * 0.01 for _ in range(staff)) if e.adopted_ai else 0
        return EmployerDecision(hire, delay, augment, layoffs, "no_preference", "rule")

    def worker(self, w: Worker, cap: float, employer_adopted: bool, quarter: str) -> WorkerDecision:
        r = self.rng.random()
        if not w.employed:
            if w.quarters_unemployed >= 2 and cap > 0.5 and r < 0.3:
                return WorkerDecision("switch_occupation", "rule")
            if r > 0.9:
                return WorkerDecision("upskill", "rule")
            return WorkerDecision("search", "rule")
        if not w.ai_adopter:
            p = 0.04 * cap + (self.c["employer_encouragement_boost"] * 0.1 * cap if employer_adopted else 0)
            if r < p:
                return WorkerDecision("adopt_ai", "rule")
        return WorkerDecision("upskill" if r > 0.98 else "stay", "rule")

    def candidate_weight(self, w: Worker, e: Employer, cap: float, prefer: str = "no_preference") -> float:
        """Rule hiring preference (`prefer` is ignored; it is set only by LLM employers). When AI automates entry-level tasks, experience counts more;
        when it augments, novices gain most (Brynjolfsson, Li & Raymond 2025)."""
        exp = min(w.experience, 20)
        auto = cap * e.automation_share if e.adopted_ai else 0.0
        aug = cap * (1 - e.automation_share) if e.adopted_ai else 0.0
        novice_bonus = self.c["novice_productivity_gain"] * aug if exp < 3 else 0.0
        return max(0.05, 1 + 0.1 * exp * auto + novice_bonus + 0.3 * w.upskilled)


# --------------------------------------------------------------------------- LLM

EMPLOYER_SYSTEM = ("You are the hiring manager for a group of US employers. Each quarter you decide what to "
                   "do with open positions and current staff. Decide as a real employer would, given costs, "
                   "risks and what you know about AI tools. Reply only with JSON.")
WORKER_SYSTEM = ("You are a US worker making a career decision this quarter. Decide as this person "
                 "realistically would, given their age, situation and what they know about AI. Reply only with JSON.")


class LLMPolicy(RulePolicy):
    """LLM decisions; inherits the rule hiring weights and only replaces the choices."""

    def __init__(self, cfg: dict, rng: random.Random, client: LLMClient, max_retries: int = 2,
                 parallel: int = 1):
        super().__init__(cfg, rng)
        self.client, self.max_retries = client, max_retries
        self.parallel = max(1, int(parallel))  # concurrent requests for worker decisions
        self.name = f"llm:{client.name}"

    def _ask(self, system: str, prompt: str, seed: int) -> tuple[dict | None, str]:
        raw = ""
        for attempt in range(self.max_retries + 1):
            try:
                if self.client.mode == "chat":
                    raw = self.client.generate(prompt, system=system, seed=seed * 10 + attempt)
                else:  # base model: put the instructions in the document and let it continue the JSON
                    raw = "{" + self.client.generate(f"{system}\n\n{prompt}\nAnswer:\n{{", seed=seed * 10 + attempt)
            except Exception as e:  # timeout or server error: count as an invalid attempt
                raw = f"<error: {type(e).__name__}>"
            obj = parse_json(raw)
            if obj is not None:
                return obj, raw
        return None, raw

    def employer(self, e: Employer, cap: float, vacancies: int, staff: int, applicants: dict,
                 adopted_share: float, quarter: str) -> EmployerDecision:
        prompt = (
            f"Quarter: {quarter}\n"
            f"Occupation group: {e.major_title} (exposure of its tasks to generative AI: {_level(e.exposure)}).\n"
            f"Current capability of AI tools for this work: {_level(cap)} ({cap:.0%}). "
            f"About {e.automation_share:.0%} of AI use in this work automates tasks; the rest assists workers.\n"
            f"Your group {'already uses' if e.adopted_ai else 'does not yet use'} AI tools; "
            f"{adopted_share:.0%} of other employer groups do.\n"
            f"Current staff: {staff}. Open positions to decide on: {vacancies}.\n"
            f"Applicants: {applicants.get('young', 0)} recent graduates aged 22-25, "
            f"{applicants.get('experienced', 0)} experienced workers.\n"
            f"Recent events: {'; '.join(e.memory) or 'none'}.\n\n"
            "For the open positions, decide how many to hire now, how many to delay, and how many to cover "
            "with AI tools instead of hiring (they must add up to the number of open positions). Also decide how "
            "many current staff to lay off, and whether you prefer experienced or junior candidates.\n"
            'JSON: {"hire": 0, "delay": 0, "augment": 0, "layoffs": 0, '
            '"prefer": "experienced|junior|no_preference", "adopt_ai": true, "reason": "<one sentence>"}'
        )
        obj, raw = self._ask(EMPLOYER_SYSTEM, prompt, seed=self.rng.randrange(10**6))
        if obj is None:
            d = super().employer(e, cap, vacancies, staff, applicants, adopted_share, quarter)
            d.valid, d.raw, d.reason = False, raw, "fallback to rule: invalid JSON"
            return d
        counts = [max(0, int(_num(obj.get(k)))) for k in EMPLOYER_ACTIONS]
        counts = _rescale(counts, vacancies)
        if isinstance(obj.get("adopt_ai"), bool):
            e.adopted_ai = e.adopted_ai or obj["adopt_ai"]
        elif counts[2] > 0:
            e.adopted_ai = True
        prefer = str(obj.get("prefer", "no_preference")).lower()
        prefer = prefer if prefer in ("experienced", "junior") else "no_preference"
        layoffs = min(staff, max(0, int(_num(obj.get("layoffs")))))
        return EmployerDecision(*counts, layoffs, prefer, str(obj.get("reason", ""))[:300], raw)

    def worker(self, w: Worker, cap: float, employer_adopted: bool, quarter: str) -> WorkerDecision:
        return self.workers_batch([(w, cap, employer_adopted, quarter)])[0]

    def workers_batch(self, items: list[tuple[Worker, float, bool, str]]) -> list[WorkerDecision]:
        """Decide for many workers at once, sending up to `parallel` requests concurrently.

        Seeds are drawn from the policy RNG in list order before any request is sent and
        rule fallbacks run afterwards in list order, so the result does not depend on
        `parallel` (Ollama answers are fixed by the seed).
        """
        prompts = [self._worker_prompt(*it) for it in items]
        seeds = [self.rng.randrange(10**6) for _ in items]
        jobs = list(zip(prompts, seeds))
        if self.parallel > 1 and len(jobs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.parallel) as pool:
                answers = list(pool.map(lambda j: self._ask(WORKER_SYSTEM, j[0][0], j[1]), jobs))
        else:
            answers = [self._ask(WORKER_SYSTEM, p, s) for (p, _), s in jobs]
        out = []
        for (w, cap, employer_adopted, quarter), (_, actions), (obj, raw) in zip(items, prompts, answers):
            action = str((obj or {}).get("action", "")).strip().lower()
            if action not in actions:
                d = super().worker(w, cap, employer_adopted, quarter)
                d.valid, d.raw = False, raw
            else:
                d = WorkerDecision(action, str(obj.get("reason", ""))[:300], raw)
            out.append(d)
        return out

    def _worker_prompt(self, w: Worker, cap: float, employer_adopted: bool, quarter: str) -> tuple[str, tuple]:
        actions = EMPLOYED_ACTIONS if w.employed else UNEMPLOYED_ACTIONS
        status = ("employed" if w.employed else f"unemployed for {w.quarters_unemployed} quarter(s)")
        prompt = (
            f"Quarter: {quarter}\n"
            f"You are {int(w.age)} years old, trained as {w.title.lower()}, with {w.experience:.0f} years of experience. "
            f"You are {status}. You {'already use' if w.ai_adopter else 'do not use'} AI tools at work.\n"
            f"AI tools can now do a {_level(cap)} share ({cap:.0%}) of the tasks in your occupation. "
            f"Employers in your field {'are using' if employer_adopted else 'are not yet using'} AI.\n"
            f"Recent events: {'; '.join(w.memory) or 'none'}.\n\n"
            f"What do you do this quarter? choose one of: {' | '.join(actions)}\n"
            'JSON: {"action": "<one of the options>", "reason": "<one sentence>"}'
        )
        return prompt, actions

    def candidate_weight(self, w: Worker, e: Employer, cap: float, prefer: str = "no_preference") -> float:
        base = 1 + 0.3 * w.upskilled
        if prefer == "experienced":
            return base * (1 + 0.15 * min(w.experience, 20))
        if prefer == "junior":
            return base * (2.0 if w.experience < 3 else 1.0)
        return base


def _num(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _rescale(counts: list[int], total: int) -> list[int]:
    """Make hire/delay/augment add up to the number of open positions (largest remainder)."""
    s = sum(counts)
    if total == 0:
        return [0, 0, 0]
    if s == 0:
        return [total, 0, 0]
    raw = [c * total / s for c in counts]
    out = [math.floor(r) for r in raw]
    for i in sorted(range(3), key=lambda i: raw[i] - out[i], reverse=True)[: total - sum(out)]:
        out[i] += 1
    return out
