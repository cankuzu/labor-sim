"""Compare Experiment 1 agent answers with the real Pew Wave 157 respondents.

    python scripts/analyze_survey.py                       # newest results/survey/*/answers.csv
    python scripts/analyze_survey.py --answers results/survey/20261127-1800/answers.csv
    python scripts/analyze_survey.py --human-only          # just the weighted Pew shares

Writes next to answers.csv: human_distribution.csv, jsd.csv, optimism_gap.csv,
subgroup_rank_correlation.csv, and prints a per-model summary.

Human shares use WEIGHT_W157, drop refused / "don't know" answers, drop workers with
several jobs and no primary job (as Pew's published results do), and only count people
who were asked the question (e.g. AIWRKFEEL was asked only of those who had heard of AI
at work). Subgroups: everyone ("all"), age group, education and industry, matching the
fields copied into the personas.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.survey import metrics as M  # noqa: E402
from labor_sim.survey.personas import PEW_COLUMNS, PEW_EXCLUDE, is_missing  # noqa: E402
from labor_sim.survey.runner import load_questions  # noqa: E402

SUBGROUPS = {"all": None, "age_group": "age", "education": "education", "occupation_group": "occupation_group"}


def load_pew(path: Path) -> pd.DataFrame:
    import pyreadstat
    df, _ = pyreadstat.read_sav(str(path), apply_value_formats=True)
    for var, bad in PEW_EXCLUDE.items():
        if var in df:
            df = df[df[var].astype(str) != bad]
    df = df.assign(all="all")
    for name, key in SUBGROUPS.items():
        if key:
            df[name] = df[PEW_COLUMNS[key]].astype(str)
    return df


def human_table(pew: pd.DataFrame, questions: list[dict]) -> pd.DataFrame:
    w = PEW_COLUMNS["weight"]
    out = []
    for q in questions:
        labels = [o["label"] for o in q["options"]]
        col = pew[q["pew_var"]].map(lambda v: _label(v, labels))
        d = pew.assign(_ans=col).dropna(subset=["_ans"])
        for sg in SUBGROUPS:
            sub = d[~d[sg].map(is_missing)]
            out.append(M.human_distribution(sub, "_ans", q["id"], sg, w))
    return pd.concat(out, ignore_index=True)


def _label(v: object, labels: list[str]) -> str | None:
    """Pew value -> option label. Unlabelled numeric codes (AIWRKDONE2 code 5 = 'None') map by position."""
    if is_missing(v):
        return None
    s = str(v)
    if s in labels:
        return s
    try:
        i = int(float(s))
    except ValueError:
        return None
    return labels[i - 1] if 1 <= i <= len(labels) else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path)
    ap.add_argument("--pew", type=Path)
    ap.add_argument("--human-only", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    questions = load_questions(resolve(cfg["survey"]["questions_file"]))
    pew_path = a.pew or next(iter(sorted(resolve(cfg["paths"]["raw"]).joinpath("pew").glob("*.sav"))), None)
    if not pew_path:
        sys.exit("Pew .sav not found in data/raw/pew/")
    human = human_table(load_pew(pew_path), questions)

    if a.human_only:
        overall = human[human["subgroup_var"] == "all"].pivot_table(index=["question_id", "option_label"], values="share")
        print("Weighted Pew shares (all workers asked the question):")
        print((overall["share"] * 100).round(1).to_string())
        return 0

    answers = a.answers or max(resolve(cfg["paths"]["results"]).glob("survey/*/answers.csv"),
                                key=lambda p: p.stat().st_mtime, default=None)  # newest run
    if not answers:
        sys.exit("No answers.csv found; run scripts/run_survey.py first.")
    ans = pd.read_csv(answers, keep_default_na=False, na_values=[""]).assign(all="all")  # keep the answer "None"
    out = Path(answers).parent
    agent = pd.concat([M.agent_distribution(ans, sg) for sg in SUBGROUPS], ignore_index=True)
    jsd = M.jsd_table(agent, human)
    gaps = M.optimism_gap(agent, human, M.valence_map(questions))
    rho = M.subgroup_rank_correlation(gaps[gaps["subgroup_var"] != "all"])
    for name, t in [("human_distribution", human), ("jsd", jsd), ("optimism_gap", gaps),
                    ("subgroup_rank_correlation", rho)]:
        t.to_csv(out / f"{name}.csv", index=False)

    print(f"answers: {answers}")
    print("\nValid answers by model:\n" + ans.groupby("model")["valid"].mean().round(3).to_string())
    print("\nJensen-Shannon distance to Pew (all workers; 0 = same distribution):")
    print(jsd[jsd["subgroup_var"] == "all"].pivot_table(index="question_id", columns="model", values="jsd").round(3).to_string())
    print("\nOptimism gap, agent minus human mean valence (all workers; > 0 = agents more positive):")
    print(gaps[gaps["subgroup_var"] == "all"].pivot_table(index="question_id", columns="model", values="gap").round(3).to_string())
    print(f"\nSaved tables to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
