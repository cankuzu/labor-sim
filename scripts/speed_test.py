"""Time real LLM calls with the simulation's own prompts and estimate run lengths.

    python scripts/speed_test.py --model llama-instruct --calls 40

Starts a normal LLM simulation, counts and times every model call, stops after
--calls calls and prints seconds per call plus the time a full run would take.
Needs a running Ollama server (config/local.yaml: llm.backend = ollama).
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


class _Enough(Exception):
    pass


class TimedClient:
    """Wraps a client, records the duration of each call, stops after `limit` calls."""

    def __init__(self, inner, limit: int):
        self.inner, self.limit, self.times = inner, limit, []
        self.name, self.mode = inner.name, inner.mode

    def generate(self, *args, **kwargs):
        if len(self.times) >= self.limit:
            raise _Enough
        t = time.time()
        out = self.inner.generate(*args, **kwargs)
        self.times.append(time.time() - t)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-instruct")
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--workers", type=int, default=200)
    a = ap.parse_args(argv)

    cfg = load_config()
    occ = pd.read_csv(resolve(cfg["paths"]["processed"]) / "occupations.csv", dtype={"soc": str})
    sc = SimConfig.from_dict(cfg["simulation"] | {"n_workers": a.workers})
    if f"pctl_{sc.exposure_source}" not in occ:
        sc.exposure_source = "eloundou"
    client = TimedClient(make_client(a.model, cfg), a.calls)
    client.generate("Reply only with JSON: {\"ok\": true}", seed=0)  # warm-up: loads the model into GPU memory
    client.times.clear()
    policy = LLMPolicy(asdict(sc), random.Random(1), client, cfg["llm"]["max_retries"])
    try:
        Simulation(occ, sc, policy=policy).run()
    except _Enough:
        pass
    per_call = sum(client.times) / max(len(client.times), 1)
    quarters = 13
    est = {w: round((w + 90) * quarters * per_call / 3600, 2) for w in (100, 250, 500, 1000)}
    print(json.dumps({"model": a.model, "backend": cfg["llm"]["backend"], "calls": len(client.times),
                      "sec_per_call": round(per_call, 3),
                      "sec_per_call_p90": round(sorted(client.times)[int(0.9 * (len(client.times) - 1))], 3)
                      if client.times else None,
                      "hours_per_run_by_workers": est}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
