#!/bin/bash
#SBATCH --job-name=astro-p1-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00

set -euo pipefail

RUN_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
GEMMA_ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it
GEMMA_REVISION=5305c1e72ea29c01f31a81230d52b375ba88b409
GPT_DESCRIPTIONS="$RUN_ROOT/data/phase1/gpt41mini_freeform_64_v1.jsonl"
GPT_COST="$RUN_ROOT/data/phase1/gpt41mini_freeform_64_cost_v1.json"

deadline=$((SECONDS + 3 * 60 * 60))
while (( SECONDS < deadline )); do
  model_ready=false
  gpt_ready=false

  if [[ -s "$GEMMA_ROOT/model-00001-of-00002.safetensors" \
        && -s "$GEMMA_ROOT/model-00002-of-00002.safetensors" \
        && -s "$GEMMA_ROOT/REVISION" \
        && "$(cat "$GEMMA_ROOT/REVISION")" == "$GEMMA_REVISION" ]]; then
    model_ready=true
  fi

  if [[ -s "$GPT_DESCRIPTIONS" && -s "$GPT_COST" ]] \
      && [[ "$(grep -cve '^[[:space:]]*$' "$GPT_DESCRIPTIONS")" -eq 64 ]]; then
    gpt_ready=true
  fi

  echo "phase1-input-gate model_ready=$model_ready gpt_ready=$gpt_ready elapsed_seconds=$SECONDS"
  if [[ "$model_ready" == true && "$gpt_ready" == true ]]; then
    echo "PHASE1_INPUTS_READY"
    exit 0
  fi
  sleep 30
done

echo "Timed out waiting for Phase 1 model and frozen GPT artifacts" >&2
exit 1
