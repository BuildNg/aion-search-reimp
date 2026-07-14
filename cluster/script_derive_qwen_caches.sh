#!/bin/bash
#SBATCH --job-name=astro-p2-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
ENV_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge

export PYTHONNOUSERSITE=1
export PYTHONPATH="$RUN_ROOT/src"

cd "$RUN_ROOT"
"$ENV_ROOT/bin/python" -u scripts/derive_qwen_embedding_caches.py \
  --source-dir results/phase1_open_text_v2 \
  --output-dir data/cache/qwen_embeddings/fp32_normalized_v1
