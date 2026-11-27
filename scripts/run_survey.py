"""Run Experiment 1 for one or more model conditions.

    python scripts/run_survey.py --models llama-instruct llama-base --n 200 --seeds 1 2 3
    python scripts/run_survey.py --backend mock --synthetic     # dry run, no GPU, no Pew file

Writes results/survey/<timestamp>/answers.jsonl and answers.csv.

Long runs (e.g. on Colab, one seed per session, results on Google Drive):
    python scripts/run_survey.py --seeds 1 --out /content/drive/MyDrive/labor-sim-results/survey/main --resume
--resume skips answers already in <out>/answers.jsonl, so a run that was cut off continues
where it stopped; answers.csv is rebuilt from the whole log (all seeds so far).
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from labor_sim.config import load_config, resolve  # noqa: E402
from labor_sim.llm.client import make_client  # noqa: E402
from labor_sim.survey import personas as P  # noqa: E402
from labor_sim.survey.runner import load_done, load_questions, run_survey  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama-instruct", "llama-base", "mistral-instruct", "mistral-base", "qwen-instruct"])
    ap.add_argument("--n", type=int, help="personas per seed (default from config)")
    ap.add_argument("--seeds", type=int, nargs="+")
    ap.add_argument("--backend", choices=["mock", "ollama"])
    ap.add_argument("--pew", help="path to the Pew Wave 157 .sav file")
    ap.add_argument("--parallel", type=int, help="concurrent requests (default: llm.parallel_requests)")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic personas (dry runs only)")
    ap.add_argument("--out", type=Path, help="output folder (default results/survey/<timestamp>)")
    ap.add_argument("--resume", action="store_true", help="skip answers already in <out>/answers.jsonl")
    ap.add_argument("--no-shuffle", action="store_true", help="show options in Pew order (default: shuffled)")
    a = ap.parse_args(argv)

    cfg = load_config(overrides={"llm": {"backend": a.backend}} if a.backend else None)
    n, seeds = a.n or cfg["survey"]["n_personas"], a.seeds or cfg["survey"]["seeds"]
    questions = load_questions(resolve(cfg["survey"]["questions_file"]))
    if not all(q.get("verified") for q in questions):
        print("WARNING: questions.yaml still has draft wording (verified: false). Fine for dry runs only.")

    pew = None
    if not a.synthetic:
        import pyreadstat
        path = a.pew or next(iter(sorted(resolve(cfg["paths"]["raw"]).joinpath("pew").glob("*.sav"))), None)
        if not path:
            sys.exit("Pew .sav not found in data/raw/pew/. Use --pew PATH or --synthetic for a dry run.")
        pew, _ = pyreadstat.read_sav(str(path), apply_value_formats=True)

    out = a.out or resolve(cfg["paths"]["results"]) / "survey" / dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    done = load_done(out / "answers.jsonl") if a.resume else {}
    if done:
        print(f"resume: {len(done)} answers already in {out / 'answers.jsonl'}")
    shuffle = not a.no_shuffle and cfg["survey"].get("shuffle_options", True)
    frames = []
    for seed in seeds:
        personas = P.synthetic(n, seed) if a.synthetic else P.from_pew(pew, n, seed)
        for key in a.models:
            client = make_client(key, cfg)
            todo = sum((key, seed, p.persona_id, q["id"]) not in done for p in personas for q in questions)
            print(f"seed {seed} | {key} ({client.mode}) | {n} personas x {len(questions)} questions | {todo} to ask",
                  flush=True)
            if todo:
                client.generate("Hello", seed=0, max_tokens=1)  # load the model before sending parallel requests
            frames.append(run_survey(client, personas, questions, seed, log_path=out / "answers.jsonl", condition=key,
                                     max_retries=cfg["llm"]["max_retries"],
                                     parallel=a.parallel or cfg["llm"].get("parallel_requests", 1),
                                     shuffle=shuffle, done=done))
    df = pd.concat(frames)
    log = pd.read_json(out / "answers.jsonl", lines=True) if (out / "answers.jsonl").exists() else df
    keys = ["condition", "seed", "persona_id", "question_id"]
    log = log.sort_values("valid", kind="stable").drop_duplicates(keys, keep="last")  # retried answers: keep the valid one
    log.to_csv(out / "answers.csv", index=False)
    errors = df["raw"].astype(str).str.startswith("<error").groupby(df["model"]).mean()
    if errors.any():
        print(f"\nRequests that failed (timeouts etc.) by model:\n{errors.round(3)}")
    print(f"\nValid answer rate by model:\n{df.groupby('model')['valid'].mean().round(3)}\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
