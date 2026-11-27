"""Run Experiment 1: every persona answers every question, for one model condition.

Each answer is written to a JSONL log (raw text included) so nothing has to be
re-run to re-score it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import yaml

from labor_sim.llm.client import LLMClient, parse_json, parse_letter
from labor_sim.survey.personas import Persona
from labor_sim.survey.prompts import ANSWER_SCHEMA, base_prompt, instruct_prompt, letters


def stable_seed(*parts: object) -> int:
    """Seed that is identical across runs (Python's hash() is randomised per process)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % 10**6


def load_questions(path: str | Path) -> list[dict]:
    return yaml.safe_load(Path(path).read_text())["questions"]


def _safe(fn, *args, **kwargs) -> str:
    """A timed-out or failed request counts as an invalid attempt instead of stopping the run."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # httpx timeouts, connection resets, HTTP 5xx
        return f"<error: {type(e).__name__}>"


def ask(client: LLMClient, persona: Persona, q: dict, seed: int, max_retries: int = 2) -> dict:
    valid = letters(len(q["options"]))
    raw, answer, reason = "", None, None
    for attempt in range(max_retries + 1):
        s = seed * 1000 + attempt
        if client.mode == "chat":
            system, user = instruct_prompt(persona, q)
            raw = _safe(client.generate, user, system=system, seed=s, schema=ANSWER_SCHEMA)
            obj = parse_json(raw) or {}
            a = str(obj.get("answer", "")).strip().upper()[:1]
            answer, reason = (a if a in valid else None), obj.get("reason")
        else:
            raw = _safe(client.generate, base_prompt(persona, q), seed=s, max_tokens=8, temperature=None)
            answer = parse_letter(raw, valid)
        if answer:
            break
    opt = q["options"][valid.index(answer)] if answer else None
    return {"question_id": q["id"], "answer": answer, "option_label": opt["label"] if opt else None,
            "valence": opt.get("valence") if opt else None, "reason": reason, "raw": raw,
            "valid": answer is not None, "attempts": attempt + 1}


def run_survey(client: LLMClient, personas: Iterable[Persona], questions: list[dict], seed: int,
               log_path: Path | None = None, condition: str | None = None,
               on_progress: Callable[[int], None] | None = None, max_retries: int = 2,
               parallel: int = 1) -> pd.DataFrame:
    """Every persona answers every question. `parallel` > 1 sends that many requests at once;
    each answer depends only on its own (persona, question, seed), so the result is the same."""
    personas = list(personas)
    jobs = [(p, q) for p in personas for q in questions]

    def one(job: tuple[Persona, dict]) -> dict:
        p, q = job
        return {"model": client.name, "condition": condition or client.mode, "seed": seed, **p.to_dict(),
                **ask(client, p, q, seed=stable_seed(p.persona_id, q["id"], seed), max_retries=max_retries)}

    rows: list[dict] = []
    fh = log_path.open("a") if log_path else None
    try:
        if parallel > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(parallel) as pool:
                results = pool.map(one, jobs)  # lazy, keeps job order
                for i, rec in enumerate(results):
                    rows.append(rec)
                    if fh:
                        fh.write(json.dumps(rec) + "\n")
                    if on_progress and (i + 1) % len(questions) == 0:
                        on_progress((i + 1) // len(questions))
        else:
            for i, job in enumerate(jobs):
                rec = one(job)
                rows.append(rec)
                if fh:
                    fh.write(json.dumps(rec) + "\n")
                if on_progress and (i + 1) % len(questions) == 0:
                    on_progress((i + 1) // len(questions))
    finally:
        if fh:
            fh.close()
    return pd.DataFrame(rows)
