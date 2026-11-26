"""Experiment 1 metrics: agents vs humans, per subgroup.

`human` must be a long table with columns
    question_id, subgroup_var, subgroup_value, option_label, share
(weighted shares from the Pew file; build it with human_distribution()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def agent_distribution(answers: pd.DataFrame, subgroup_var: str) -> pd.DataFrame:
    a = answers[answers["valid"]]
    d = (a.groupby(["model", "condition", "question_id", subgroup_var])["option_label"]
           .value_counts(normalize=True).rename("share").reset_index())
    return d.rename(columns={subgroup_var: "subgroup_value"}).assign(subgroup_var=subgroup_var)


def human_distribution(df: pd.DataFrame, question_var: str, question_id: str, subgroup_var: str,
                       weight_var: str, labels: dict | None = None) -> pd.DataFrame:
    x = df[[question_var, subgroup_var, weight_var]].dropna()
    if labels:
        x[question_var] = x[question_var].map(labels)
    g = x.groupby([subgroup_var, question_var])[weight_var].sum()
    share = (g / g.groupby(level=0).transform("sum")).rename("share").reset_index()
    return share.rename(columns={subgroup_var: "subgroup_value", question_var: "option_label"}).assign(
        question_id=question_id, subgroup_var=subgroup_var)


def jsd_table(agent: pd.DataFrame, human: pd.DataFrame) -> pd.DataFrame:
    """Jensen-Shannon distance (base 2, 0 = identical, 1 = disjoint) per model x question x subgroup."""
    rows = []
    keys = ["question_id", "subgroup_var", "subgroup_value"]
    for (model, cond), ag in agent.groupby(["model", "condition"]):
        for k, h in human.groupby(keys):
            a = ag[(ag[keys[0]] == k[0]) & (ag[keys[1]] == k[1]) & (ag[keys[2]].astype(str) == str(k[2]))]
            if a.empty:
                continue
            labels = sorted(set(h["option_label"]) | set(a["option_label"]))
            p = a.set_index("option_label")["share"].reindex(labels, fill_value=0).to_numpy()
            q = h.set_index("option_label")["share"].reindex(labels, fill_value=0).to_numpy()
            rows.append(dict(zip(keys, k)) | {"model": model, "condition": cond,
                                               "jsd": float(jensenshannon(p, q, base=2))})
    return pd.DataFrame(rows)


def valence_map(questions: list[dict]) -> dict[tuple[str, str], float]:
    """(question_id, option_label) -> valence, for questions whose options all carry a valence.

    Keyed by question because the same label ("Yes") means opposite things for
    "hopeful" and "worried"."""
    out = {}
    for q in questions:
        vals = [o.get("valence") for o in q["options"]]
        if all(v is not None for v in vals):
            out.update({(q["id"], o["label"]): float(o["valence"]) for o in q["options"]})
    return out


def mean_valence(dist: pd.DataFrame, valence: dict[tuple[str, str], float]) -> pd.Series:
    keys = list(zip(dist["question_id"], dist["option_label"]))
    d = dist.assign(v=[valence.get(k) for k in keys])
    d = d[d["question_id"].isin({q for q, _ in valence})]
    d["v"] = d["v"].astype(float) * d["share"]
    by = [c for c in ["model", "condition", "question_id", "subgroup_var", "subgroup_value"] if c in d]
    return d.groupby(by)["v"].sum()


def optimism_gap(agent: pd.DataFrame, human: pd.DataFrame, valence: dict[tuple[str, str], float]) -> pd.DataFrame:
    """Agent mean valence minus human mean valence (>0 = agents more positive than people).
    Build `valence` with valence_map(questions)."""
    a = mean_valence(agent, valence).rename("agent").reset_index()
    h = mean_valence(human, valence).rename("human").reset_index()
    m = a.merge(h, on=["question_id", "subgroup_var", "subgroup_value"])
    return m.assign(gap=m["agent"] - m["human"])


def subgroup_rank_correlation(gaps: pd.DataFrame) -> pd.DataFrame:
    """Do agents order subgroups (e.g. by worry) the same way humans do? Spearman rho per model x question."""
    rows = []
    for k, g in gaps.groupby(["model", "condition", "question_id", "subgroup_var"]):
        rho = spearmanr(g["agent"], g["human"]).statistic if len(g) >= 3 else np.nan
        rows.append(dict(zip(["model", "condition", "question_id", "subgroup_var"], k)) | {"spearman_rho": rho, "n": len(g)})
    return pd.DataFrame(rows)
