"""Time real LLM calls with the simulation's own prompts and estimate run lengths.

    python scripts/speed_test.py --model llama-instruct
    python scripts/speed_test.py --model llama-instruct --parallel 1 4 8

Builds a simulation, then times (a) employer-cell calls, which run one after another,
and (b) one batch of worker calls at each --parallel level, which run concurrently.
Prints effective seconds per call and the hours a full 13-quarter run would take.
Needs a running Ollama server (config/local.yaml: llm.backend = ollama) started with
OLLAMA_NUM_PARALLEL at least as large as the biggest --parallel value.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.llm.client import make_client  # noqa: E402
from labor_sim.sim.engine import SimConfig, Simulation  # noqa: E402
from labor_sim.sim.policies import LLMPolicy  # noqa: E402

QUARTERS = 13
EMPLOYER_CELLS = 90


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-instruct")
    ap.add_argument("--employer-calls", type=int, default=12)
    ap.add_argument("--worker-calls", type=int, default=48)
    ap.add_argument("--parallel", type=int, nargs="+", default=[1, 4])
    a = ap.parse_args(argv)

    cfg = load_config()
    occ = pd.read_csv(resolve(cfg["paths"]["processed"]) / "occupations.csv", dtype={"soc": str})
    sc = SimConfig.from_dict(cfg["simulation"] | {"n_workers": max(200, a.worker_calls)})
    if f"pctl_{sc.exposure_source}" not in occ:
        sc.exposure_source = "eloundou"
    client = make_client(a.model, cfg)
    t = time.time()
    client.generate('Reply only with JSON: {"ok": true}', seed=0)  # loads the model into GPU memory
    warmup = time.time() - t

    sim = Simulation(occ, sc)
    q, step = sim.quarters[6], 6
    policy = LLMPolicy(asdict(sc), random.Random(1), client, cfg["llm"]["max_retries"])

    cells = list(sim.employers.items())[: a.employer_calls]
    t = time.time()
    for cell, e in cells:
        policy.employer(e, sim.capability(e, step), 3, 20, {"young": 2, "experienced": 3}, 0.3, q)
    employer_s = (time.time() - t) / max(len(cells), 1)

    workers = sim.workers[: a.worker_calls]
    items = [(w, sim.capability(sim.employers[w.cell], step), True, q) for w in workers]
    worker_s = {}
    for p in a.parallel:
        policy.parallel = p
        t = time.time()
        decisions = policy.workers_batch(items)
        worker_s[p] = (time.time() - t) / len(items)
        valid = sum(getattr(d, "valid", True) for d in decisions) / len(decisions)
        print(f"parallel={p}: {worker_s[p]:.2f} s per worker call, valid answers {valid:.0%}", flush=True)

    best = min(worker_s, key=worker_s.get)
    est = {}
    for n in (100, 250, 500, 1000):
        hours = QUARTERS * (EMPLOYER_CELLS * employer_s + n * worker_s[best]) / 3600
        est[n] = round(hours, 2)
    print(json.dumps({"model": a.model, "backend": cfg["llm"]["backend"], "warmup_s": round(warmup, 1),
                      "employer_sec_per_call": round(employer_s, 2),
                      "worker_sec_per_call": {k: round(v, 2) for k, v in worker_s.items()},
                      "best_parallel": best,
                      "hours_per_run_by_workers": est}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
