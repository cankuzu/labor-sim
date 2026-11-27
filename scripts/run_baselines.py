"""Random Forest baseline on the occupation table: predict each occupation's 2022->2025
employment change (log) from AI exposure features, and compare with a naive predictor.

    python scripts/run_baselines.py
Writes results/baselines/random_forest.json and predictions.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

from labor_sim.baselines import random_forest as RF  # noqa: E402
from labor_sim.config import load_config, resolve  # noqa: E402


def main() -> int:
    cfg = load_config()
    occ = pd.read_csv(resolve(cfg["paths"]["processed"]) / "occupations.csv", dtype={"soc": str})
    out = resolve(cfg["paths"]["results"]) / "baselines"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    # automation_share exists for only about half of the occupations, so run with and without it
    for name, drop in {"all_features": [], "without_automation_share": ["automation_share"]}.items():
        RF.DEFAULT_FEATURES = [f for f in RF.DEFAULT_FEATURES if f not in drop]
        res = RF.evaluate(occ)
        y = res["predictions"]["observed"]
        res["naive_mae"] = float(mean_absolute_error(y, [y.median()] * len(y)))
        res["predictions"].to_csv(out / f"predictions_{name}.csv", index=False)
        summary[name] = {k: v for k, v in res.items() if k != "predictions"}
        print(f"{name}: n={res['n']} MAE={res['mae']:.3f} (naive {res['naive_mae']:.3f}) "
              f"Spearman={res['spearman']:.3f}")
        if name == "all_features":
            RF.DEFAULT_FEATURES = list(RF.DEFAULT_FEATURES) + ["automation_share"]
    d = occ.dropna(subset=["emp_change_2022_2025"])
    summary["exposure_vs_employment_change_spearman"] = {
        s: float(spearmanr(x[f"exposure_{s}"], x["emp_change_2022_2025"]).statistic)
        for s in ["anthropic", "eloundou", "microsoft", "aioe"] if f"exposure_{s}" in d
        for x in [d.dropna(subset=[f"exposure_{s}"])]}
    (out / "random_forest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["exposure_vs_employment_change_spearman"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
