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
