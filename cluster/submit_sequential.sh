#!/bin/bash

set -euo pipefail

ROOT=/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search
cd "$ROOT"

wait_id=$(sbatch --parsable --output="$ROOT/logs/slurm-%j-wait.out" cluster/script_wait_artifacts.sh)
phase0_id=$(sbatch --parsable --dependency="afterok:$wait_id" --output="$ROOT/logs/slurm-%j-phase0.out" cluster/script_phase0.sh)
phase1_id=$(sbatch --parsable --dependency="afterok:$phase0_id" --output="$ROOT/logs/slurm-%j-phase1.out" cluster/script_phase1.sh)

printf 'wait_artifacts_job=%s\nphase0_job=%s\nphase1_job=%s\n' "$wait_id" "$phase0_id" "$phase1_id"
