# AION-Search open reimplementation

Independent, config-driven reimplementation of the AION-Search alignment and retrieval pipeline. The authors' checkout under `orig_repo/` is read-only reference material.

## Current scope

Phase 0/1 preparation is implemented locally. Model and dataset commands are review-gated and run only on the THQL A100 cluster.

```text
configs/              pinned Phase 0/1 definitions
cluster/              sequential Slurm launch chain
data/prompts/         frozen caption schema
data/queries/         review-gated canonical queries and paraphrases
src/aion_reimp/       pipeline modules
tests/                synthetic contracts; cluster tests are marked
orig_repo/            authors' release, unchanged
```

The `src/aion_reimp/` package boundary is intentional: it prevents accidental imports from the repository root, gives scripts and tests one stable namespace, and supports the installed `aion-reimp` CLI without adding an application framework.

## Local checks

```powershell
$env:PYTHONPATH='src'
python -m aion_reimp.cli validate-config configs/phase0_reference.yaml
python -m aion_reimp.cli validate-config configs/phase1_open_text.yaml
python -m pytest -m "not cluster"
```

Local checks do not load model weights. The one-time R-OAI query freeze is the exception: it runs locally through OpenRouter using `OPEN_ROUTER_KEY`, and only its Parquet and metadata sidecar are synced. The key must never be placed on the cluster. Reference equivalence, evaluation, Qwen captioning, and Qwen embedding run on THQL only after review.

When work is stopped for review, the gate is after local code and tests are ready but before any GitHub push, cluster sync, or cluster run. The reviewer should be able to inspect the exact prompt, queries, config, and proposed commands at that point.

## Cluster location

- project: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search`
- uv environment: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge`
- shared Hugging Face cache: `/data2/cmdir/home/ioit_thql/.cache/huggingface`

Do not blanket-sync this repository. Sync only the reviewed files, preserve the existing Slurm resource lines, and record the job ID and output paths.

The approved launch chain submits an artifact-completeness gate, Phase 0 after that gate, and Phase 1 after a successful Phase 0. GPU jobs run offline against exact cached revisions; closed-API keys are explicitly removed from the Slurm environment.
