# AION-Search open reimplementation

Independent, config-driven reimplementation of the AION-Search alignment and retrieval pipeline. The authors' checkout under `orig_repo/` is read-only reference material.

## Current scope

Phase 0 is complete. Phase 1 reuses the frozen Qwen and GPT descriptions, applies the released GalaxyBench judge prompt and Pydantic schema through XGrammar-constrained Gemma decoding, and reports released human-path overlap as its primary metric. Phase 2 remains unlaunched.

```text
configs/              pinned Phase 0/1 definitions and Phase 2 smoke contract
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
python -m aion_reimp.cli validate-config configs/phase1.yaml
python -m aion_reimp.cli validate-config configs/phase2_smoke.yaml
python -m pytest -m "not cluster"
```

Before a Phase 2 launch, build and inspect `data/gates/phase2_launch_contract_v1.json`. The contract records the passed Phase 0 reference gate and carries Phase 1 caption metrics as diagnostic evidence. Phase 1 caption quality does not manually block the engineering smoke run; only a failed Phase 0 reproducibility gate does.

Local checks do not load model weights. The frozen Qwen3-VL and GPT-4.1-mini descriptions are judged by the same text-only Gemma model on THQL. XGrammar enforces the released GalaxyBench response schema during decoding; `OPEN_ROUTER_KEY` never moves to the cluster.

When work is stopped for review, the gate is after local code and tests are ready but before any GitHub push, cluster sync, or cluster run. The reviewer should be able to inspect the exact prompt, queries, config, and proposed commands at that point.

The active Phase-6 crossmatch scale run is review-gated. It expands the authoritative 3,602-row HSC feasibility subset to exactly 18,000 HSC coordinates from the same pinned image source, after the same benchmark exclusions. It queries only coordinate/redshift/quality columns from the pinned DESI HATS catalog:

```bash
python scripts/run_phase6_crossmatch_cluster.py --preflight
python scripts/run_phase6_crossmatch_cluster.py
```

The first command creates only `preflight/phase6_hsc_crossmatch_18k_v1.json`. The second refuses to create `results/phase6_hsc_crossmatch_18k_v1/` unless the exact source fingerprint and preflight contract still pass. Neither command loads images, captions, embeddings, spectrum arrays, or model weights. The selected-match artifact uses the locked 1-arcsecond radius; the summary reports whether at least 1,000 quality-valid pairs were found.

The completed HSC scale crossmatch yielded 1,576 clean pairs. The paired redshift run freezes those exact rows and compares image-only, spectrum-only, and image+spectrum recovery. Preparation is CPU-only so the DESI stream scan does not occupy a GPU; analysis then reuses the existing AION spectrum adapter, deterministic splits, standardized ridge fitting, and spec-z metrics:

```bash
python scripts/run_phase6_paired_redshift_cluster.py preflight
python scripts/run_phase6_paired_redshift_cluster.py prepare
python scripts/run_phase6_paired_redshift_cluster.py analyze
```

The two CPU-only checks before joint retrieval reuse that completed run:

```bash
python scripts/run_phase6_prechecks.py alpha-sensitivity
python scripts/run_phase6_prechecks.py galaxy-zoo
```

The first command refits the saved embeddings with the wider ridge grid; it
does no encoder inference. The second requires the checksum-pinned public
Galaxy Zoo DESI friendly Parquet named in `configs/phase6_prechecks.yaml` at
the configured local path. It reads only coordinates and required vote
fractions, then writes coverage and redshift-support tables. Neither command
trains or evaluates a retrieval model.

## Cluster location

- project: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search`
- uv environment: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge`
- shared Hugging Face cache: `/data2/cmdir/home/ioit_thql/.cache/huggingface`
- Gemma 4 model: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it`

Do not blanket-sync this repository. Sync only the reviewed files, preserve the existing Slurm resource lines, and record the job ID and output paths.

The approved launch chain submits an artifact-completeness gate, Phase 0 after that gate, and Phase 1 after a successful Phase 0. GPU jobs run offline against exact cached revisions; closed-API keys are explicitly removed from the Slurm environment.
