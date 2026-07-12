#!/bin/bash
#SBATCH --job-name=astro-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:20:00

set -euo pipefail

ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
LOG="$ROOT/logs/artifact_download.log"
CACHE=/data2/cmdir/home/ioit_thql/.cache/huggingface/hub

echo "Waiting for the already-running pinned-artifact queue: $LOG"
until grep -Fq "DONE astronolan/galaxy-descriptions" "$LOG"; do
  sleep 300
done

required=(
  "$CACHE/models--astronolan--aion-search/snapshots/e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"
  "$CACHE/datasets--astronolan--galaxy-description-benchmark/snapshots/ebb13986d04b6b5e47529fb1fc68761839bffd75"
  "$CACHE/datasets--astronolan--gz-decals-embeddings/snapshots/c11f7a02aa1ed00b85f3dd43c222271046445a2e"
  "$CACHE/datasets--astronolan--lens-retrieval-ls-embeddings/snapshots/f5507c433552084e2b3d195a27dae5110037d64d"
  "$CACHE/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
  "$CACHE/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
  "$CACHE/datasets--astronolan--galaxy-descriptions/snapshots/6890dc0c8fc793867c07a6ce4fce11e51c167d6e"
)

for path in "${required[@]}"; do
  test -d "$path" || { echo "Required pinned snapshot is absent: $path" >&2; exit 1; }
done

echo "All pinned artifacts are complete."
