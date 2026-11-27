"""Run Experiment 2 from the command line.

    python scripts/run_simulation.py --policy rule --seeds 1 2 3
    python scripts/run_simulation.py --policy llm --model llama-instruct --workers 100   # pilot
    python scripts/run_simulation.py --policy llm --model mistral-base --exposure eloundou

Each run writes results/simulations/<run_id>/round_XX.json, summary.json,
metrics.csv (age group x exposure quintile per quarter) and cells.csv.
"""
from __future__ import annotations

import argparse
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
from labor_sim.sim.policies import LLMPolicy, RulePolicy  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["rule", "llm"], default="rule")
    ap.add_argument("--model", default="llama-instruct", help="model key from config (llm policy)")
    ap.add_argument("--backend", choices=["mock", "ollama"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--exposure", choices=["anthropic", "eloundou", "microsoft", "aioe"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    a = ap.parse_args(argv)

    cfg = load_config(overrides={"llm": {"backend": a.backend}} if a.backend else None)
    occ = pd.read_csv(resolve(cfg["paths"]["processed"]) / "occupations.csv", dtype={"soc": str})
    for seed in a.seeds:
        sc = SimConfig.from_dict(cfg["simulation"] | {"seed": seed})
        if a.workers:
            sc.n_workers = a.workers
        if a.exposure:
            sc.exposure_source = a.exposure
        if f"pctl_{sc.exposure_source}" not in occ:
            print(f"exposure '{sc.exposure_source}' not available yet; falling back to 'eloundou'")
            sc.exposure_source = "eloundou"
        rng = random.Random(seed)
        policy = (LLMPolicy(asdict(sc), rng, make_client(a.model, cfg), cfg["llm"]["max_retries"], cfg["llm"].get("parallel_requests", 1))
                  if a.policy == "llm" else RulePolicy(asdict(sc), rng))
        tag = a.model if a.policy == "llm" else "rule"
        run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}_{sc.exposure_source}_s{seed}"
        out = resolve(cfg["paths"]["results"]) / "simulations" / run_id
        sim = Simulation(occ, sc, policy=policy, run_id=run_id, out_dir=out)
        t0 = time.time()
        res = sim.run(on_round=lambda r: print(f"  {r['quarter']}: employment {r['employment_rate']:.3f}"))
        res.metrics_frame().to_csv(out / "metrics.csv", index=False)
        res.cells_frame().to_csv(out / "cells.csv", index=False)
        print(f"{run_id}: {res.status} in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
