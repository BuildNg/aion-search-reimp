#!/bin/bash
#SBATCH --job-name=astro-cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3-00:20:00

set -euo pipefail

HF_HOME=/data2/cmdir/home/ioit_thql/.cache/huggingface
CACHE="$HF_HOME/hub"
HF=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge/bin/hf

required=(
  "model|astronolan/aion-search|e6d56ee28b6768f4e3e4494b2c0b32a00abb2594"
  "dataset|astronolan/galaxy-description-benchmark|ebb13986d04b6b5e47529fb1fc68761839bffd75"
  "dataset|astronolan/gz-decals-embeddings|c11f7a02aa1ed00b85f3dd43c222271046445a2e"
  "dataset|astronolan/lens-retrieval-ls-embeddings|f5507c433552084e2b3d195a27dae5110037d64d"
  "model|Qwen/Qwen3-Embedding-0.6B|97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
  "model|Qwen/Qwen3-VL-8B-Instruct|0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
  "dataset|astronolan/galaxy-descriptions|6890dc0c8fc793867c07a6ce4fce11e51c167d6e"
)

echo "Waiting for all pinned artifacts to pass offline cache verification."
until (
  for spec in "${required[@]}"; do
    IFS='|' read -r repo_type repo revision <<< "$spec"
    HF_HUB_OFFLINE=1 HF_HOME="$HF_HOME" "$HF" download "$repo" \
      --type "$repo_type" --revision "$revision" --cache-dir "$CACHE" --quiet \
      >/dev/null || exit 1
  done
); do
  sleep 300
done

echo "All pinned artifacts are complete."
