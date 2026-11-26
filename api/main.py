"""FastAPI service for the labor market simulator.

    uvicorn api.main:app --reload
    open http://localhost:8000/docs

Long runs execute in a background thread; results are written per round to
results/ and kept in memory for the lifetime of the server.
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
import uuid
from dataclasses import asdict
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from api.schemas import Metrics, Round, RunStatus, SimulationRequest, SurveyRequest
from labor_sim.config import load_config, resolve
from labor_sim.llm.client import make_client
from labor_sim.sim.engine import SimConfig, Simulation, quarters
from labor_sim.sim.policies import LLMPolicy, RulePolicy
from labor_sim.survey import personas as P
from labor_sim.survey.runner import load_questions, run_survey

app = FastAPI(title="Labor market simulation with local LLM agents",
              description="Experiments 1 and 2 of the graduation project, as an API.", version="0.1.0")

RUNS: dict[str, dict] = {}       # id -> {"status": RunStatus, "rounds": [...], "answers": DataFrame}
LOCK = threading.Lock()


@lru_cache
def cfg() -> dict:
    return load_config()


@lru_cache
def occupations() -> pd.DataFrame:
    path = resolve(cfg()["paths"]["processed"]) / "occupations.csv"
    if not path.exists():
        raise HTTPException(503, "occupations.csv missing: run scripts/download_data.py and scripts/build_occupations.py")
    return pd.read_csv(path, dtype={"soc": str})


def _new_run(kind: str, config: dict, total: int | None) -> str:
    rid = f"{kind[:3]}-{uuid.uuid4().hex[:8]}"
    with LOCK:
        RUNS[rid] = {"status": RunStatus(id=rid, kind=kind, status="queued", config=config,
                                         rounds_total=total, created=time.time()), "rounds": []}
    return rid


def _get(rid: str) -> dict:
    run = RUNS.get(rid)
    if not run:
        raise HTTPException(404, f"run {rid} not found")
    return run


def _start(target, *args) -> None:
    threading.Thread(target=target, args=args, daemon=True).start()


# ------------------------------------------------------------------ Experiment 1

def _survey_job(rid: str, req: SurveyRequest) -> None:
    run = RUNS[rid]
    run["status"].status = "running"
    try:
        c = cfg()
        qs = load_questions(resolve(c["survey"]["questions_file"]))
        if req.synthetic_personas:
            personas = P.synthetic(req.n_personas, req.seed)
        else:
            import pyreadstat
            path = next(iter(sorted(resolve(c["paths"]["raw"]).joinpath("pew").glob("*.sav"))), None)
            if not path:
                raise FileNotFoundError("Pew .sav not in data/raw/pew/ (or set synthetic_personas=true)")
            personas = P.from_pew(pyreadstat.read_sav(str(path), apply_value_formats=True)[0], req.n_personas, req.seed)
        out = resolve(c["paths"]["results"]) / "survey" / rid
        out.mkdir(parents=True, exist_ok=True)

        def progress(i: int) -> None:
            run["status"].rounds_done = i
        df = run_survey(make_client(req.model, c), personas, qs, req.seed, log_path=out / "answers.jsonl",
                        condition=req.model, on_progress=progress, max_retries=c["llm"]["max_retries"])
        df.to_csv(out / "answers.csv", index=False)
        run["answers"] = df
        run["status"].status = "finished"
    except Exception as exc:
        run["status"].status, run["status"].error = "failed", repr(exc)
    finally:
        run["status"].finished = time.time()


@app.post("/experiments/survey", response_model=RunStatus, status_code=202, tags=["Experiment 1"])
def start_survey(req: SurveyRequest) -> RunStatus:
    """Run Experiment 1 for one model condition and persona sample."""
    rid = _new_run("survey", req.model_dump(), req.n_personas)
    _start(_survey_job, rid, req)
    return RUNS[rid]["status"]


@app.get("/experiments/survey/{rid}/answers", tags=["Experiment 1"])
def survey_answers(rid: str, limit: int = 200) -> list[dict]:
    run = _get(rid)
    if "answers" not in run:
        raise HTTPException(409, f"run is {run['status'].status}")
    return run["answers"].drop(columns=["raw"]).head(limit).replace({np.nan: None}).to_dict("records")


# ------------------------------------------------------------------ Experiment 2

def _simulation_job(rid: str, req: SimulationRequest) -> None:
    run = RUNS[rid]
    run["status"].status = "running"
    try:
        c = cfg()
        sc = SimConfig.from_dict(c["simulation"] | req.model_dump(exclude={"policy", "model"}))
        rng = random.Random(sc.seed)
        policy = (LLMPolicy(asdict(sc), rng, make_client(req.model, c), c["llm"]["max_retries"])
                  if req.policy == "llm" else RulePolicy(asdict(sc), rng))
        out = resolve(c["paths"]["results"]) / "simulations" / rid
        sim = Simulation(occupations(), sc, policy=policy, run_id=rid, out_dir=out)

        def on_round(r: dict) -> None:
            run["rounds"].append(r)
            run["status"].rounds_done = len(run["rounds"])
        res = sim.run(on_round=on_round)
        res.metrics_frame().to_csv(out / "metrics.csv", index=False)
        res.cells_frame().to_csv(out / "cells.csv", index=False)
        run["status"].status = "finished"
    except Exception as exc:
        run["status"].status, run["status"].error = "failed", repr(exc)
    finally:
        run["status"].finished = time.time()


@app.post("/simulations", response_model=RunStatus, status_code=202, tags=["Experiment 2"])
def start_simulation(req: SimulationRequest) -> RunStatus:
    """Start an Experiment 2 run. Returns at once with an id; poll /simulations/{id} or open the websocket."""
    occ = occupations()
    if f"pctl_{req.exposure_source}" not in occ:
        raise HTTPException(422, f"exposure source '{req.exposure_source}' not in occupations.csv yet")
    rid = _new_run("simulation", req.model_dump(), len(quarters(req.start_quarter, req.end_quarter)))
    _start(_simulation_job, rid, req)
    return RUNS[rid]["status"]


@app.get("/simulations", response_model=list[RunStatus], tags=["Experiment 2"])
def list_simulations() -> list[RunStatus]:
    return [r["status"] for r in RUNS.values() if r["status"].kind == "simulation"]


@app.get("/simulations/{rid}", response_model=RunStatus, tags=["Experiment 2"])
def simulation_status(rid: str) -> RunStatus:
    """Run status and configuration."""
    return _get(rid)["status"]


@app.get("/simulations/{rid}/rounds/{n}", response_model=Round, tags=["Experiment 2"])
def simulation_round(rid: str, n: int) -> dict:
    """All employer-cell decisions and group outcomes for one quarter."""
    rounds = _get(rid)["rounds"]
    if not 0 <= n < len(rounds):
        raise HTTPException(404, f"round {n} not available ({len(rounds)} done)")
    return rounds[n]


@app.get("/simulations/{rid}/metrics", response_model=Metrics, tags=["Experiment 2"])
def simulation_metrics(rid: str) -> Metrics:
    """Employment, hires and separations by age group x exposure quintile, plus the headline young-worker gap."""
    run = _get(rid)
    rows = [g | {"quarter": r["quarter"]} for r in run["rounds"] for g in r["groups"]]
    if not rows:
        return Metrics(id=rid, quarters=[], by_group=[])
    df = pd.DataFrame(rows)
    return Metrics(id=rid, quarters=list(dict.fromkeys(df["quarter"])),
                   by_group=df.replace({np.nan: None}).to_dict("records"),
                   young_exposed_gap=young_exposed_gap(df))


def young_exposed_gap(df: pd.DataFrame) -> dict | None:
    """Difference-in-differences of employment rates: (young - older) x (high - low exposure), first vs last quarter."""
    d = df.assign(exp=np.where(df["exposure_quintile"] >= 4, "high", np.where(df["exposure_quintile"] <= 2, "low", "mid")))
    d = d[d["exp"] != "mid"]
    qs = list(dict.fromkeys(d["quarter"]))
    if len(qs) < 2:
        return None
    s = d[d["quarter"].isin([qs[0], qs[-1]])].groupby(["quarter", "age_group", "exp"])[["employed", "labor_force"]].sum()
    rate = (s["employed"] / s["labor_force"]).unstack("quarter")
    change = (rate[qs[-1]] - rate[qs[0]]).unstack("exp")
    young = [a for a in change.index if a.startswith("22")][0]
    older = [a for a in change.index if a != young][0]
    did = (change.loc[young, "high"] - change.loc[young, "low"]) - (change.loc[older, "high"] - change.loc[older, "low"])
    return {"from": qs[0], "to": qs[-1], "change": change.round(4).to_dict("index"), "diff_in_diff": round(float(did), 4)}


@app.websocket("/simulations/{rid}/live")
async def simulation_live(ws: WebSocket, rid: str) -> None:
    """Streams each finished quarter as JSON while the run is going, then closes."""
    await ws.accept()
    sent = 0
    try:
        while True:
            run = RUNS.get(rid)
            if not run:
                await ws.send_json({"error": f"run {rid} not found"})
                break
            rounds = run["rounds"]
            while sent < len(rounds):
                r = rounds[sent]
                await ws.send_json({"round": r["round"], "quarter": r["quarter"],
                                    "employment_rate": r["employment_rate"], "groups": r["groups"]})
                sent += 1
            if run["status"].status in ("finished", "failed") and sent >= len(rounds):
                await ws.send_json({"status": run["status"].status, "error": run["status"].error})
                break
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return
    await ws.close()


@app.get("/health", tags=["Service"])
def health() -> dict:
    c = cfg()
    return {"ok": True, "llm_backend": c["llm"]["backend"], "models": list(c["llm"]["models"])}
