#!/usr/bin/env bash
# One-shot setup on a Colab / Kaggle GPU machine (no Google Drive needed).
#   !git clone -q https://github.com/cankuzu/labor-sim.git /content/labor-sim; bash /content/labor-sim/scripts/colab_bootstrap.sh
# Optional: MODELS="llama3.1:8b-instruct-q4_K_M qwen2.5:7b-instruct-q4_K_M" bash ...
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS="${MODELS:-llama3.1:8b-instruct-q4_K_M}"

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"

echo "== Ollama"
if ! command -v ollama >/dev/null; then
  apt-get -qq install -y zstd >/dev/null 2>&1 || true
  curl -fsSL https://ollama.com/install.sh | sh >/tmp/ollama_install.log 2>&1
fi
if ! curl -s localhost:11434/api/version >/dev/null; then
  setsid nohup ollama serve >/tmp/ollama.log 2>&1 < /dev/null &
  for _ in $(seq 30); do curl -s localhost:11434/api/version >/dev/null && break; sleep 1; done
fi
curl -s localhost:11434/api/version; echo
for m in $MODELS; do echo "pull $m"; ollama pull "$m" >/tmp/pull.log 2>&1 && echo "  ok" || tail -3 /tmp/pull.log; done
ollama list

echo "== Python packages and data"
pip install -q -r requirements.txt
printf 'llm:\n  backend: ollama\n  ollama_url: http://localhost:11434\n' > config/local.yaml
python scripts/download_data.py --only github crosswalk anthropic >/tmp/download.log 2>&1 || tail -5 /tmp/download.log
python scripts/build_occupations.py | tail -4
echo "== ready"
