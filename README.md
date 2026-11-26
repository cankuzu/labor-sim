# labor-sim

**Simulating the Labor Market Impact of Generative AI with Local LLM Agents**
Graduation project: Tahir Can Kuzu, Efe Yılmaz, Muhammed Kerim Erdoğan.

Do locally run LLM agents reproduce how workers and the labor market respond to
generative AI, and how much of the gap comes from instruction tuning? The code
runs two experiments on open-weight models (base vs instruct) and serves them
through a FastAPI service.

| Part | What it does | Code |
| --- | --- | --- |
| Data | Downloads 9 open datasets and merges them on 6-digit SOC codes | `scripts/download_data.py`, `labor_sim/data/` |
| Experiment 1 | LLM personas answer the Pew Wave 157 workplace-AI questions; compared with real workers | `labor_sim/survey/` |
| Experiment 2 | Quarterly agent-based labor market 2022Q4–2025Q4 with LLM employers and workers | `labor_sim/sim/` |
| Baselines | Rule-based simulation (Bass diffusion, retraining paths) and a Random Forest | `labor_sim/sim/policies.py`, `labor_sim/baselines/` |
| API | Start runs, read rounds and metrics, live websocket | `api/` |

## Quick start (no GPU needed)

```bash
pip install -r requirements.txt
python scripts/download_data.py --only github   # 2 minutes, no accounts
python scripts/build_occupations.py
python -m pytest -q                              # 16 tests

# dry runs with the mock LLM
python scripts/run_survey.py --backend mock --synthetic --n 30 --seeds 1
python scripts/run_simulation.py --policy rule --seeds 1 2 3
python scripts/run_simulation.py --policy llm --backend mock --model llama-instruct --workers 200

# API
uvicorn api.main:app --reload        # then open http://localhost:8000/docs
```

## Running with real models

1. Start an Ollama server: on your PC, or on a free Colab GPU with
   [notebooks/colab_setup.ipynb](https://colab.research.google.com/github/cankuzu/labor-sim/blob/main/notebooks/colab_setup.ipynb)
   (opens directly in Colab; it clones this repo, installs Ollama and runs a speed test).
2. Pull the four models and check their tags in `config/default.yaml`
   (`ollama pull llama3.1:8b-instruct-q4_K_M`, ...). If a base model has no Ollama tag,
   import its GGUF from Hugging Face with a `Modelfile`.
3. Create `config/local.yaml`:
   ```yaml
   llm:
     backend: ollama
     ollama_url: http://localhost:11434
   ```
4. Pilot first: `python scripts/run_simulation.py --policy llm --model llama-instruct --workers 100`
   and time it. A full run is roughly (workers + ~90 employer cells) × 13 quarters calls,
   about 14,000 for 1,000 workers. Raise `llm_worker_decisions_every` to save time.

## Before results count

- [x] Pew questions filled in from the Wave 157 topline (AIWRKFEEL_a-d, AIWRKOPPO, CHATWRK, AIWRKDONE2)
- [ ] ...then compared once with the PDF and codebook by a person, and `verified: true` set
- [ ] `PEW_COLUMNS` confirmed in the Wave 157 codebook (current values are Pew's usual names; occupation item still TODO)
- [ ] OEWS May 2022–2025 downloaded (the 2021 file is a development stand-in)
- [x] `load_anthropic()` matched to the current Hugging Face layout (uses `labor_market_impacts/job_exposure.csv`)
- [ ] Anthropic Economic Index downloaded (`python scripts/download_data.py --only anthropic`)
- [ ] SOC 2010→2018 crosswalk downloaded so AIOE matches all occupations
- [ ] IPUMS CPS extract and Census→SOC crosswalk in place, `build_targets.py` re-run
- [ ] `indeed_sector_map.csv` reviewed (sector → SOC prefixes is a manual mapping)
- [ ] Calibration values in `config/default.yaml` checked against the papers cited next to them

## Design notes

- **Employer cells.** Employers are grouped as SOC major group × exposure quintile, so each
  cell has enough workers for one decision per quarter (about 90 cells).
- **AI capability** in a cell follows a logistic path over the 13 quarters, scaled by the
  cell's exposure percentile from the chosen source (`exposure_source`).
- **Validation.** The headline number is a difference-in-differences:
  (young − older) × (high − low exposure) change in employment, first to last quarter,
  returned by `GET /simulations/{id}/metrics` and comparable to Brynjolfsson et al. (2025)
  and to the IPUMS CPS targets.
- **Invalid LLM answers** are retried (`max_retries`); if still invalid, the rule decision is
  used and the record is flagged `valid: false`, so the invalid-answer rate can be reported.
- Every run writes one JSON per quarter plus `metrics.csv` and `cells.csv` under `results/`.

## Layout

```
config/            default.yaml (all parameters), local.yaml (your overrides, not committed)
scripts/           download_data, build_occupations, build_targets, run_survey, run_simulation
labor_sim/data/    occupation merge, validation targets, Indeed sector map
labor_sim/llm/     Ollama client, mock client, JSON parsing
labor_sim/survey/  personas, questions, prompts, runner, metrics
labor_sim/sim/     agents, policies (rule and LLM), engine
labor_sim/baselines/ Random Forest
api/               FastAPI app and schemas
notebooks/         Colab / Kaggle setup
tests/             pytest
```
