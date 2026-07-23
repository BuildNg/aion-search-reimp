#!/bin/bash
#SBATCH --job-name=astro-p6-distance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -u scripts/run_phase6_multimodal_retrieval.py distance
