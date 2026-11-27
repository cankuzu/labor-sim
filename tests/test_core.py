import random
from dataclasses import asdict

import pandas as pd

from labor_sim.data.occupations import soc6
from labor_sim.llm.client import MockClient, parse_json, parse_letter
from labor_sim.sim.engine import SimConfig, Simulation, quarters
from labor_sim.sim.policies import LLMPolicy, _rescale
from labor_sim.survey.personas import synthetic
from labor_sim.survey.runner import load_questions, run_survey
from labor_sim.config import resolve


def test_soc6():
    assert soc6("15-1252.00") == "15-1252"
    assert soc6("151252") == "15-1252"
    assert soc6("") == ""


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"action": "search", "reason": "x"} Hope that helps.')["action"] == "search"
    assert parse_json('{{"hire": 2}') == {"hire": 2}
    assert parse_json("no json here") is None


def test_parse_letter():
    assert parse_letter("B) Somewhat worried", ["A", "B", "C"]) == "B"
    assert parse_letter("(C)", ["A", "B", "C"]) == "C"
    assert parse_letter("I think D) maybe", ["A", "B", "C"]) is None
    assert parse_letter("A) Yes\n\nInterviewer:", ["A", "B"]) == "A"
    assert parse_letter("2.\n\nInterviewer:", ["A", "B", "C"]) == "B"
    assert parse_letter("9.", ["A", "B"]) is None
    assert parse_letter("Yes, I feel overwhelmed.", ["A", "B"], ["Yes", "No"]) == "A"
    assert parse_letter("Not much, really", ["A", "B", "C"], ["Not sure", "Not much", "None"]) == "B"
    assert parse_letter("Absolutely", ["A", "B"]) is None
    assert parse_letter("A few times a week.", list("ABC"), ["Every day", "A few times a week", "Never"]) == "B"
    assert parse_letter("C) Never", list("ABC"), ["Every day", "A few times a week", "Never"]) == "C"


def test_rescale_sums_to_total():
    for counts, total in [([1, 1, 1], 5), ([0, 0, 0], 3), ([10, 0, 0], 2), ([2, 3, 4], 0)]:
        assert sum(_rescale(counts, total)) == total


def test_quarters():
    q = quarters("2022Q4", "2025Q4")
    assert q[0] == "2022Q4" and q[-1] == "2025Q4" and len(q) == 13


def test_survey_mock_all_valid():
    qs = load_questions(resolve("labor_sim/survey/questions.yaml"))
    df = pd.concat([run_survey(MockClient("m-chat", "chat"), synthetic(10, 1), qs, 1),
                    run_survey(MockClient("m-base", "completion"), synthetic(10, 1), qs, 1)])
    assert df["valid"].all() and len(df) == 2 * 10 * len(qs)


def test_survey_reproducible():
    qs = load_questions(resolve("labor_sim/survey/questions.yaml"))
    a = run_survey(MockClient("m", "chat"), synthetic(5, 2), qs, 2)["answer"].tolist()
    b = run_survey(MockClient("m", "chat"), synthetic(5, 2), qs, 2)["answer"].tolist()
    assert a == b


def test_optimism_gap_uses_question_specific_valence():
    from labor_sim.survey import metrics as M
    qs = load_questions(resolve("labor_sim/survey/questions.yaml"))
    val = M.valence_map(qs)
    assert val[("feel_hopeful", "Yes")] == 1 and val[("feel_worried", "Yes")] == -1
    assert not any(q == "chatbot_use" for q, _ in val)
    human = pd.DataFrame({"question_id": ["feel_worried"] * 2, "subgroup_var": "all", "subgroup_value": "all",
                          "option_label": ["Yes", "No"], "share": [0.6, 0.4]})
    agent = human.assign(model="m", condition="chat", share=[0.2, 0.8])
    gap = M.optimism_gap(agent, human, val)
    assert abs(gap["gap"].iloc[0] - 0.4) < 1e-9   # agents worry less -> positive gap


def test_rule_simulation_runs(occ):
    res = Simulation(occ, SimConfig(n_workers=300, seed=1)).run()
    assert res.status == "finished" and len(res.rounds) == 13
    m = res.metrics_frame()
    assert set(m["age_group"]) == {"22-25", "26+"}
    assert m["employment_rate"].between(0, 1).all()


def test_rule_simulation_reproducible(occ):
    a = Simulation(occ, SimConfig(n_workers=200, seed=7)).run().rounds[-1]["employment_rate"]
    b = Simulation(occ, SimConfig(n_workers=200, seed=7)).run().rounds[-1]["employment_rate"]
    assert a == b


def test_llm_simulation_mock_chat_and_base(occ):
    for mode in ("chat", "completion"):
        sc = SimConfig(n_workers=150, seed=2)
        pol = LLMPolicy(asdict(sc), random.Random(2), MockClient(f"mock-{mode}", mode))
        res = Simulation(occ, sc, policy=pol).run()
        cells = res.cells_frame().dropna(subset=["valid"])
        assert res.status == "finished"
        assert cells["valid"].all()
        assert ((cells["hire"] + cells["delay"] + cells["augment"]) >= 0).all()


def test_no_ai_keeps_employment_flat(occ):
    """Sanity check: with almost no AI capability the economy should not drift."""
    rates = [Simulation(occ, SimConfig(n_workers=600, seed=s, ai_capability_growth=1e-6)).run()
             for s in range(3)]
    drift = sum(r.rounds[-1]["employment_rate"] - r.rounds[0]["employment_rate"] for r in rates) / 3
    assert abs(drift) < 0.03


def test_parallel_worker_calls_match_sequential(occ):
    """Concurrent worker requests must give the same run as one-at-a-time requests."""
    runs = []
    for parallel in (1, 4):
        sc = SimConfig(n_workers=120, seed=3)
        pol = LLMPolicy(asdict(sc), random.Random(3), MockClient("mock", "chat"), parallel=parallel)
        res = Simulation(occ, sc, policy=pol).run()
        runs.append(([r["employment_rate"] for r in res.rounds], res.cells_frame()))
    assert runs[0][0] == runs[1][0]
    assert runs[0][1].equals(runs[1][1])  # employer decisions too (NaN-safe comparison)


def test_parallel_survey_matches_sequential():
    qs = load_questions(resolve("labor_sim/survey/questions.yaml"))
    a = run_survey(MockClient("m", "chat"), synthetic(8, 4), qs, 1)
    b = run_survey(MockClient("m", "chat"), synthetic(8, 4), qs, 1, parallel=4)
    assert a["answer"].tolist() == b["answer"].tolist() and a["persona_id"].tolist() == b["persona_id"].tolist()


def _fake_pew(n=60, seed=0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    pick = lambda opts: rng.choice(opts, n)
    return pd.DataFrame({
        "F_AGECAT": pick(["18-29", "30-49", "50-64", "65+", "Refused"]),
        "F_GENDER": pick(["A man", "A woman"]),
        "F_EDUCCAT2": pick(["High school graduate", "Postgraduate"]),
        "F_INC_SDT1": pick(["Less than $30,000", "$100,000 or more"]),
        "INDUSTRYCOMBO_W157": pick(["Education", "Other"]),
        "JOBNUM_W157": pick(["One", "More than one and do not consider any of them my primary job"]),
        "WEIGHT_W157": rng.uniform(0.2, 4.0, n),
        "AIWRKFEEL_c_W157": pick(["Yes", "No", "Don't know/Refused/Web blank", None]),
        "AIWRKDONE2_W157": pick(["All", "Some", 5.0, "Not sure"]),
    })


def test_pew_personas_skip_refused_and_multi_job():
    from labor_sim.survey.personas import from_pew, pew_eligible
    df = _fake_pew()
    ok = pew_eligible(df)
    assert not (ok["F_AGECAT"] == "Refused").any()
    assert (ok["JOBNUM_W157"] == "One").all()
    ps = from_pew(df, 30, seed=1)
    assert all(18 <= p.age <= 75 and p.gender in ("man", "woman") for p in ps)
    assert "a field not listed" in next(p.backstory() for p in ps if p.occupation_group == "Other")


def test_pew_human_shares():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("an", Path(__file__).parent.parent / "scripts" / "analyze_survey.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    df = _fake_pew().assign(all="all")
    for name, key in an.SUBGROUPS.items():
        if key:
            df[name] = df[an.PEW_COLUMNS[key]].astype(str)
    qs = [{"id": "feel_worried", "pew_var": "AIWRKFEEL_c_W157", "options": [{"label": "Yes"}, {"label": "No"}]},
          {"id": "done", "pew_var": "AIWRKDONE2_W157",
           "options": [{"label": x} for x in ["All", "Most", "Some", "Not much", "None", "Not sure"]]}]
    h = an.human_table(df, qs)
    tot = h.groupby(["question_id", "subgroup_var", "subgroup_value"])["share"].sum()
    assert (tot.round(6) == 1).all()
    assert set(h[h.question_id == "feel_worried"]["option_label"]) <= {"Yes", "No"}
    assert "None" in set(h[h.question_id == "done"]["option_label"])
    assert "Refused" not in set(h[h.subgroup_var == "age_group"]["subgroup_value"])


def test_option_order_and_mapping():
    from labor_sim.survey.runner import ask, option_order
    q = {"id": "q", "text": "?", "options": [{"label": "More"}, {"label": "Fewer"}, {"label": "Same"},
                                            {"label": "Not sure", "fixed": True}]}
    orders = {tuple(option_order(q, s)) for s in range(40)}
    assert len(orders) > 1 and all(o[-1] == 3 for o in orders)
    ordq = dict(q, order="ordinal")
    assert {tuple(option_order(ordq, s)) for s in range(40)} == {(0, 1, 2, 3), (2, 1, 0, 3)}

    class Picker:  # always answers the option whose text is "Fewer", wherever it is shown
        name, mode = "picker", "chat"
        def generate(self, prompt, system=None, **kw):
            line = next(l for l in prompt.splitlines() if l.endswith(") Fewer"))
            return '{"answer": "%s", "reason": "x"}' % line[0]
    from labor_sim.survey.personas import synthetic
    p = synthetic(1)[0]
    for s in range(10):
        r = ask(Picker(), p, q, seed=s, shuffle=True)
        assert r["option_label"] == "Fewer"


def test_survey_resume_skips_done(tmp_path):
    from labor_sim.survey.runner import load_done, load_questions, run_survey
    from labor_sim.survey.personas import synthetic
    qs = load_questions("labor_sim/survey/questions.yaml")
    ps = synthetic(3, seed=1)
    log = tmp_path / "answers.jsonl"
    calls = {"n": 0}
    base = MockClient(name="m")
    class Counting:
        name, mode = "m", "chat"
        def generate(self, *a, **k):
            calls["n"] += 1
            return base.generate(*a, **k)
    first = run_survey(Counting(), ps[:2], qs, seed=1, log_path=log, condition="m", shuffle=True)
    n1 = calls["n"]
    done = load_done(log)
    again = run_survey(Counting(), ps, qs, seed=1, log_path=log, condition="m", shuffle=True, done=done)
    assert calls["n"] - n1 == len(qs)          # only the new persona was asked
    assert len(again) == 3 * len(qs)
    assert again.iloc[: len(first)]["option_label"].tolist() == first["option_label"].tolist()


def test_answer_none_survives_csv(tmp_path):
    import pandas as pd
    p = tmp_path / "a.csv"
    pd.DataFrame({"option_label": ["None", "Some", None]}).to_csv(p, index=False)
    d = pd.read_csv(p, keep_default_na=False, na_values=[""])
    assert d["option_label"].tolist()[:2] == ["None", "Some"] and pd.isna(d["option_label"].iloc[2])
