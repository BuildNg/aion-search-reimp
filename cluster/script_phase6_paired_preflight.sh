#!/bin/bash
#SBATCH --job-name=astro-p6-pair-check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -u scripts/run_phase6_paired_redshift_cluster.py preflight
