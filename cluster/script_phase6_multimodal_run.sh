#!/bin/bash
#SBATCH --job-name=astro-p6-mm-retrieval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00

set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found" >&2
  exit 1
fi

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
export TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -u scripts/run_phase6_multimodal_retrieval.py joint
