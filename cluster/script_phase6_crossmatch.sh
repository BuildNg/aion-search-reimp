#!/bin/bash
#SBATCH --job-name=astro-p6-hsc18k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
# Submit only after reviewing the preflight report. The Python entrypoint
# independently rejects a stale report before creating the run directory.
"$ENV_ROOT/bin/python" -u scripts/run_phase6_crossmatch_cluster.py
