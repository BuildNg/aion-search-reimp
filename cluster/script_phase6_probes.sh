#!/bin/bash
#SBATCH --job-name=astro-p6-probes
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
# Phase 6 deliberately does NOT set HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE: the
# Multimodal Universe DESI sample is streaming-only by design (no bulk
# download), so this job needs live network access to Hugging Face, unlike
# the fully offline Phase 0-3 jobs.
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"
export TOKENIZERS_PARALLELISM=false
unset OPENAI_API_KEY OPEN_ROUTER_KEY || true

cd "$RUN_ROOT"
# Safe first-cluster-contact: this job performs preflight only. Inspect the
# report and resolve every VERIFY-ON-CLUSTER item before submitting the
# separate full-run launcher, cluster/script_phase6_probes_full.sh.
"$ENV_ROOT/bin/python" -u scripts/run_phase6_probes_cluster.py --preflight
