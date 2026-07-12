#!/bin/bash
#SBATCH --job-name=astro-p0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:20:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -u scripts/run_phase0_cluster.py --config configs/phase0_reference.yaml
