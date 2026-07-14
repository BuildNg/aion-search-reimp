#!/bin/bash
#SBATCH --job-name=astro-p1n-diag
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:20:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge
GEMMA_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it
GEMMA_REVISION=5305c1e72ea29c01f31a81230d52b375ba88b409

test -s "$GEMMA_ROOT/model-00001-of-00002.safetensors"
test -s "$GEMMA_ROOT/model-00002-of-00002.safetensors"
test "$(cat "$GEMMA_ROOT/REVISION")" = "$GEMMA_REVISION"

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
export TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -c 'import pydantic, xgrammar; assert pydantic.VERSION.startswith("2.")'
"$ENV_ROOT/bin/python" -u scripts/run_phase1_extractor_diagnostic.py \
  --config configs/phase1_nested.yaml \
  --output-dir results/phase1_diag_nested_v1
