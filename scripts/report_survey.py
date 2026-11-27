"""Final Experiment 1 tables and figures, with uncertainty.

    python scripts/report_survey.py --dir /content/drive/MyDrive/labor-sim-results/survey/main

Needs <dir>/answers.csv (run_survey.py) and <dir>/human_distribution.csv (analyze_survey.py).
Writes <dir>/report/:
  summary_by_model.csv    valid rate, mean JSD, mean optimism gap per model, with 95% CIs
  per_question.csv        agent vs Pew share for every option, JSD and gap per question
  base_vs_instruct.csv    instruct minus base for each model family, with 95% CIs
  seeds.csv               the same headline numbers per seed (stability check)
  subgroups.csv           do agents rank subgroups (age, education, industry) like Pew does?
  optimism_gap.png, yes_shares.png

CIs: personas are resampled within each seed (1,000 bootstrap draws); the Pew shares are
held fixed (they come from 5,273 weighted respondents). Base and instruct models answered
the same personas in each seed, so their difference uses a paired bootstrap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.survey import metrics as M  # noqa: E402
from labor_sim.survey.runner import load_questions  # noqa: E402

FAMILIES = {"llama": ("llama-base", "llama-instruct"), "mistral": ("mistral-base", "mistral-instruct")}
FEELINGS = ["feel_hopeful", "feel_excited", "feel_worried", "feel_overwhelmed"]


def model_arrays(ans: pd.DataFrame, questions: list[dict]) -> dict:
    """Per model: persona keys (seed, persona_id) and, per question, the chosen option index (-1 = invalid)."""
    out = {}
    for m, g in ans.groupby("condition"):
        wide = g.pivot_table(index=["seed", "persona_id"], columns="question_id", values="option_label", aggfunc="first")
        arr = {}
        for q in questions:
            labels = [o["label"] for o in q["options"]]
            col = wide[q["id"]] if q["id"] in wide else pd.Series(index=wide.index, dtype=object)
            arr[q["id"]] = col.map(lambda v: labels.index(v) if v in labels else -1).to_numpy()
        out[m] = (wide.index.to_frame(index=False), arr)
    return out


def stats(arr: dict, idx: np.ndarray, questions: list[dict], human: dict, hval: dict) -> dict:
    """JSD per question, optimism gap per valenced question, for the personas in idx."""
    r = {}
    for q in questions:
        a = arr[q["id"]][idx]
        a = a[a >= 0]
        k = len(q["options"])
        p = np.bincount(a, minlength=k) / max(len(a), 1)
        r[("jsd", q["id"])] = float(jensenshannon(p, human[q["id"]], base=2)) if len(a) else np.nan
        vals = [o.get("valence") for o in q["options"]]
        if all(v is not None for v in vals):
            r[("gap", q["id"])] = float(p @ np.array(vals, float) - hval[q["id"]]) if len(a) else np.nan
    gaps = [r[("gap", q)] for q in FEELINGS if ("gap", q) in r]
    r[("gap", "feelings_mean")] = float(np.mean(gaps))
    r[("gap", "all_mean")] = float(np.nanmean([v for (k, _), v in r.items() if k == "gap" and _ != "feelings_mean"]))
    r[("jsd", "all_mean")] = float(np.nanmean([v for (k, q), v in r.items() if k == "jsd" and q != "all_mean"]))
    r[("jsd", "feelings_mean")] = float(np.mean([r[("jsd", q)] for q in FEELINGS]))
    return r


def boot_idx(keys: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Resample personas within each seed."""
    parts = [rng.choice(np.flatnonzero(keys["seed"].to_numpy() == s), size=(keys["seed"] == s).sum())
             for s in sorted(keys["seed"].unique())]
    return np.concatenate(parts)


def ci(x: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    cfg = load_config()
    questions = load_questions(resolve(cfg["survey"]["questions_file"]))
    ans = pd.read_csv(a.dir / "answers.csv")
    hum = pd.read_csv(a.dir / "human_distribution.csv")
    out = a.dir / "report"
    out.mkdir(exist_ok=True)

    hall = hum[hum["subgroup_var"] == "all"]
    human, hval = {}, {}
    for q in questions:
        labels = [o["label"] for o in q["options"]]
        h = hall[hall["question_id"] == q["id"]].set_index("option_label")["share"].reindex(labels, fill_value=0)
        human[q["id"]] = h.to_numpy() / h.sum()
        vals = [o.get("valence") for o in q["options"]]
        if all(v is not None for v in vals):
            hval[q["id"]] = float(human[q["id"]] @ np.array(vals, float))

    arrays = model_arrays(ans, questions)
    rng = np.random.default_rng(0)
    point, boots = {}, {}
    common = None
    for m, (keys, arr) in arrays.items():
        n = len(keys)
        point[m] = stats(arr, np.arange(n), questions, human, hval)
        boots[m] = []
        common = keys if common is None else common
    # one set of bootstrap draws reused for every model, so base/instruct differences are paired
    keyset = {m: arrays[m][0].assign(i=range(len(arrays[m][0]))).set_index(["seed", "persona_id"])["i"] for m in arrays}
    ref = next(iter(arrays.values()))[0]
    for _ in range(a.boot):
        draw = ref.iloc[boot_idx(ref, rng)]
        for m in arrays:
            idx = keyset[m].reindex(pd.MultiIndex.from_frame(draw)).dropna().astype(int).to_numpy()
            boots[m].append(stats(arrays[m][1], idx, questions, human, hval))

    rows = []
    for m in arrays:
        b = pd.DataFrame(boots[m])
        r = {"model": m, "n_answers": int((ans["condition"] == m).sum()),
             "valid_rate": float(ans.loc[ans["condition"] == m, "valid"].mean())}
        for key, name in [(("jsd", "all_mean"), "jsd_all"), (("jsd", "feelings_mean"), "jsd_feelings"),
                          (("gap", "feelings_mean"), "gap_feelings"), (("gap", "all_mean"), "gap_all")]:
            lo, hi = ci(b[key].to_numpy())
            r |= {name: point[m][key], f"{name}_lo": lo, f"{name}_hi": hi}
        rows.append(r)
    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(out / "summary_by_model.csv", index=False)

    perq = []
    for q in questions:
        labels = [o["label"] for o in q["options"]]
        for m, (keys, arr) in arrays.items():
            x = arr[q["id"]]
            x = x[x >= 0]
            share = np.bincount(x, minlength=len(labels)) / max(len(x), 1)
            for i, lab in enumerate(labels):
                perq.append({"question_id": q["id"], "option": lab, "model": m, "agent_share": share[i],
                             "pew_share": human[q["id"]][i], "jsd": point[m][("jsd", q["id"])],
                             "gap": point[m].get(("gap", q["id"]), np.nan)})
    pd.DataFrame(perq).to_csv(out / "per_question.csv", index=False)

    diff = []
    for fam, (base, inst) in FAMILIES.items():
        if base not in arrays or inst not in arrays:
            continue
        for key, name in [(("gap", "feelings_mean"), "gap_feelings"), (("gap", "all_mean"), "gap_all"),
                          (("jsd", "feelings_mean"), "jsd_feelings"), (("jsd", "all_mean"), "jsd_all")] + \
                         [(("gap", q), f"gap_{q}") for q in hval]:
            d = np.array([bi[key] - bb[key] for bi, bb in zip(boots[inst], boots[base])])
            lo, hi = ci(d)
            diff.append({"family": fam, "metric": name, "instruct_minus_base": point[inst][key] - point[base][key],
                         "ci_lo": lo, "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(diff).to_csv(out / "base_vs_instruct.csv", index=False)

    seeds = []
    for m, (keys, arr) in arrays.items():
        for s in sorted(keys["seed"].unique()):
            st = stats(arr, np.flatnonzero(keys["seed"].to_numpy() == s), questions, human, hval)
            seeds.append({"model": m, "seed": s, "jsd_all": st[("jsd", "all_mean")],
                          "gap_feelings": st[("gap", "feelings_mean")], "gap_all": st[("gap", "all_mean")]})
    pd.DataFrame(seeds).to_csv(out / "seeds.csv", index=False)

    sub = pd.read_csv(a.dir / "subgroup_rank_correlation.csv") if (a.dir / "subgroup_rank_correlation.csv").exists() else None
    if sub is not None and len(sub):
        s = sub.groupby(["condition", "subgroup_var"])["spearman_rho"].mean().unstack().round(3)
        s.to_csv(out / "subgroups.csv")

    _figures(summary, pd.DataFrame(perq), out)

    pd.set_option("display.width", 160)
    cols = ["model", "valid_rate", "jsd_all", "jsd_all_lo", "jsd_all_hi", "gap_feelings", "gap_feelings_lo", "gap_feelings_hi"]
    print("Per model (95% CI from persona bootstrap):")
    print(summary[cols].round(3).to_string(index=False))
    print("\nInstruct minus base (paired bootstrap):")
    print(pd.DataFrame(diff).round(3).to_string(index=False))
    print("\nPer seed:")
    print(pd.DataFrame(seeds).pivot_table(index="model", columns="seed", values="gap_feelings").round(3).to_string())
    if sub is not None and len(sub):
        print("\nMean Spearman rho, agent vs Pew subgroup ordering:")
        print(s.to_string())
    print(f"\nSaved to {out}")
    return 0


def _figures(summary: pd.DataFrame, perq: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [m for m in ["llama-base", "llama-instruct", "mistral-base", "mistral-instruct", "qwen-instruct"]
             if m in set(summary["model"])]
    s = summary.set_index("model").loc[order]
    colors = ["#8c8c8c" if "base" in m else "#3366cc" for m in order]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = s["gap_feelings"]
    ax.bar(order, y, color=colors, yerr=[y - s["gap_feelings_lo"], s["gap_feelings_hi"] - y], capsize=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Optimism gap vs Pew\n(agent − human, feelings items)")
    ax.set_title("Instruct models are more positive about AI at work than real workers")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out / "optimism_gap.png", dpi=200)
    plt.close(fig)

    yes = perq[(perq["question_id"].isin(FEELINGS)) & (perq["option"] == "Yes")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    qs = FEELINGS
    for i, m in enumerate(order):
        d = yes[yes["model"] == m].set_index("question_id").reindex(qs)
        ax.scatter(np.arange(len(qs)) + (i - 2) * 0.1, d["agent_share"] * 100, label=m, color=colors[i],
                   marker="o" if "base" not in m else "s", s=30)
    pew = yes.drop_duplicates("question_id").set_index("question_id").reindex(qs)["pew_share"] * 100
    ax.scatter(np.arange(len(qs)), pew, marker="_", s=600, color="black", label="Pew (real workers)")
    ax.set_xticks(range(len(qs)), [q.replace("feel_", "") for q in qs])
    ax.set_ylabel("% answering Yes")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "yes_shares.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
