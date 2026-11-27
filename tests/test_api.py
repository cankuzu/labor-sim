import time

from fastapi.testclient import TestClient

import api.main as main


def _client(occ, monkeypatch):
    monkeypatch.setattr(main, "occupations", lambda: occ)
    return TestClient(main.app)


def _wait(client, rid, path="/simulations/{}", timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(path.format(rid)).json()
        if s["status"] in ("finished", "failed"):
            return s
        time.sleep(0.1)
    raise TimeoutError


def test_health(occ, monkeypatch):
    assert _client(occ, monkeypatch).get("/health").json()["ok"]


def test_simulation_flow(occ, monkeypatch):
    c = _client(occ, monkeypatch)
    r = c.post("/simulations", json={"policy": "rule", "n_workers": 200, "seed": 1})
    assert r.status_code == 202
    rid = r.json()["id"]
    s = _wait(c, rid)
    assert s["status"] == "finished" and s["rounds_done"] == 13
    rnd = c.get(f"/simulations/{rid}/rounds/0").json()
    assert rnd["quarter"] == "2022Q4" and rnd["groups"]
    m = c.get(f"/simulations/{rid}/metrics").json()
    assert m["young_exposed_gap"]["diff_in_diff"] is not None
    assert c.get(f"/simulations/{rid}/rounds/99").status_code == 404
    assert any(x["id"] == rid for x in c.get("/simulations").json())


def test_llm_simulation_and_websocket(occ, monkeypatch):
    c = _client(occ, monkeypatch)
    rid = c.post("/simulations", json={"policy": "llm", "model": "llama-instruct", "n_workers": 100}).json()["id"]
    with c.websocket_connect(f"/simulations/{rid}/live") as ws:
        msgs = []
        while True:
            m = ws.receive_json()
            msgs.append(m)
            if "status" in m:
                break
    assert msgs[-1]["status"] == "finished"
    assert sum("quarter" in m for m in msgs) == 13


def test_unknown_exposure_rejected(occ, monkeypatch):
    c = _client(occ, monkeypatch)
    assert c.post("/simulations", json={"exposure_source": "anthropic"}).status_code == 422


def test_survey_flow(occ, monkeypatch):
    c = _client(occ, monkeypatch)
    rid = c.post("/experiments/survey", json={"model": "mistral-base", "n_personas": 5,
                                              "synthetic_personas": True}).json()["id"]
    s = _wait(c, rid)
    assert s["status"] == "finished", s
    rows = c.get(f"/experiments/survey/{rid}/answers").json()
    assert rows and all(r["valid"] for r in rows)
