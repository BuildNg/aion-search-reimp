#!/bin/bash
#SBATCH --job-name=astro-p1n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:20:00

set -euo pipefail

if [ -x /usr/bin/nvidia-smi ]; then
  /usr/bin/nvidia-smi
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found" >&2
  exit 1
fi

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge
GEMMA_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it
GEMMA_REVISION=5305c1e72ea29c01f31a81230d52b375ba88b409

test -s "$GEMMA_ROOT/model-00001-of-00002.safetensors"
test -s "$GEMMA_ROOT/model-00002-of-00002.safetensors"
test "$(cat "$GEMMA_ROOT/REVISION")" = "$GEMMA_REVISION"
test "$(grep -cve '^[[:space:]]*$' "$RUN_ROOT/data/phase1/gpt41mini_freeform_64_full_v2.jsonl")" -eq 64
test "$(grep -cve '^[[:space:]]*$' "$RUN_ROOT/data/phase1/qwen3vl8b_freeform_64_full_v2.jsonl")" -eq 64
test -s "$RUN_ROOT/data/phase1/gpt41mini_freeform_64_cost_v1.json"

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
export TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -c 'import pydantic, xgrammar; assert pydantic.VERSION.startswith("2.")'
"$ENV_ROOT/bin/python" -u scripts/run_phase1_cluster.py --config configs/phase1_nested.yaml
