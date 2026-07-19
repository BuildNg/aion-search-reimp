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

The Phase-6 crossmatch feasibility probe is prepared but review-gated. It reuses the completed Phase-3 10k caption manifest and queries only coordinate/redshift/quality columns from the pinned DESI HATS catalog:

```bash
python scripts/run_phase6_crossmatch_cluster.py --preflight
python scripts/run_phase6_crossmatch_cluster.py
```

The first command creates only `preflight/phase6_crossmatch_probe_v3.json`. The second refuses to create `results/phase6_crossmatch_probe_v3/` unless the exact preflight still passes. Neither command downloads spectrum arrays or loads a model. Run ID `v1` is preserved as the first-cluster-contact write failure. Run ID `v2` completed and exposed a quality-contract mismatch: the pinned HATS catalog stores raw `ZWARN` semantics (`False` means zero warnings), while the streaming probe adapter uses an inverted "no problem" boolean. Run ID `v3` keeps those interfaces separate and makes preflight require a quality-valid self-match.

## Cluster location

- project: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search`
- uv environment: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge`
- shared Hugging Face cache: `/data2/cmdir/home/ioit_thql/.cache/huggingface`
- Gemma 4 model: `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it`

Do not blanket-sync this repository. Sync only the reviewed files, preserve the existing Slurm resource lines, and record the job ID and output paths.

The approved launch chain submits an artifact-completeness gate, Phase 0 after that gate, and Phase 1 after a successful Phase 0. GPU jobs run offline against exact cached revisions; closed-API keys are explicitly removed from the Slurm environment.
