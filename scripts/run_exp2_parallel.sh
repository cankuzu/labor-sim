#!/usr/bin/env bash
# Experiment 2 on ONE GPU machine, three seeds at the same time.
#   OUT=/content/drive/MyDrive/labor-sim-results/simulations bash scripts/run_exp2_parallel.sh
#
# Models run one after another; for each model the three seeds run as three processes at once,
# so only one model is in GPU memory and Ollama batches their requests together. Ollama is
# restarted with 3 x 8 = 24 parallel slots (KV cache for 24 x 4k tokens + an 8B model fits a 24 GB L4).
# Results are identical to running the seeds one by one (fixed seeds). Finished runs are skipped,
# so the script can be started again after a disconnect. Logs: /content/logs/<model>_s<seed>.log
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="${OUT:?set OUT to the results folder}"
SEEDS="${SEEDS:-1 2 3}"
MODELS="${MODELS_KEYS:-llama-instruct llama-base mistral-instruct mistral-base qwen-instruct}"
LOGS="${LOGS:-/content/logs}"
mkdir -p "$LOGS" "$OUT"

n=$(echo $SEEDS | wc -w)
pkill -x ollama 2>/dev/null; sleep 2
OLLAMA_NUM_PARALLEL=$((8 * n)) OLLAMA_MAX_LOADED_MODELS=1 setsid nohup ollama serve >/tmp/ollama.log 2>&1 < /dev/null &
for _ in $(seq 30); do curl -s localhost:11434/api/version >/dev/null && break; sleep 1; done

for m in $MODELS; do
  echo "$(date +%H:%M:%S) $m: seeds $SEEDS" >> "$LOGS/progress.log"
  for s in $SEEDS; do
    python scripts/run_simulation.py --policy llm --models "$m" --seeds "$s" --out-root "$OUT" \
      > "$LOGS/${m}_s${s}.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M:%S) $m: done" >> "$LOGS/progress.log"
done
echo "$(date +%H:%M:%S) ALL DONE" >> "$LOGS/progress.log"
