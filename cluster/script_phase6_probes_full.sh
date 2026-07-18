#!/bin/bash
#SBATCH --job-name=astro-p6-probes-full
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00

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

export HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
export TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
# Submit only after manually reviewing the preflight-only job's report.
# The Python entrypoint independently rejects a stale report whose config,
# code, packages, or resolved device differ from this full-run environment.
"$ENV_ROOT/bin/python" -u scripts/run_phase6_probes_cluster.py
