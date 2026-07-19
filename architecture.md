# AION-Search reimplementation architecture

This file is the code-level source of truth for the open reimplementation. The implementation plan in the Life wiki owns scientific scope and phase gates; this document owns module boundaries, pipeline edges, config contracts, and run artifacts.

## Design goal

Keep one small, config-driven research codebase that can run the three primary conditions without forking the pipeline. Data identity, model-specific inference, alignment training, retrieval evaluation, and artifact writing have clear owners. Add only the abstraction needed by a current experiment.

The authors' release under `orig_repo/` is read-only reference material. New code lives directly in this repository root.

## Repository map

```text
AION-Search/
  architecture.md       code-level contract
  README.md              setup and current runnable commands
  pyproject.toml         package and environment definition
  configs/               small set of current YAML experiment configs
  cluster/               Slurm scripts and dependency-chain submission
  scripts/               data preparation, cache generation, Slurm, sync, pull
  src/aion_reimp/        reusable pipeline code
  src/spec_probes/        bounded Phase 6 frozen spectrum-encoder probes (see below)
  src/spectra_crossmatch/ separate Phase 6 captioned-image × DESI feasibility probe
  tests/                 identity, metric, model, and artifact contracts
  data/                  manifests and small metadata; no duplicated raw datasets
  results/               local pulled run artifacts; ignored
  orig_repo/             authors' unmodified reference checkout
```

`src/aion_reimp/` is a deliberately small package boundary, not an application layer. It prevents repository-root import ambiguity and gives tests, scripts, and the installed CLI one stable namespace; flattening the modules directly into `src/` would save one directory but lose those safeguards.

Only current runnable configs and launch helpers remain live. A completed run preserves its resolved config and manifests in its result directory; exploratory variants do not become permanent source branches.

## Module ownership

| Module | Owns | Must not own |
|---|---|---|
| `config.py` | strict schema, defaults, cross-field validation, resolved config | model calls, training loops |
| `manifest.py` | stable object IDs, benchmark exclusions, SHA-256 split, common-row selection, fingerprints | captioning, embeddings, optimization |
| `datasets.py` | loading manifest-selected image/text vectors and batches | split derivation, metric policy |
| `captioning.py` | matched free-form Qwen3-VL and pinned OpenRouter GPT description generation | morphology extraction, dataset joins, audit scoring |
| `morphology.py` | released Galaxy Zoo Pydantic schema and prompt, XGrammar-constrained Gemma judging, path construction, synthetic calibration | image access, caption generation, audit scoring |
| `caption_audit.py` | released human-path overlap, secondary per-question diagnostics, paired bootstraps | caption generation, extraction, retrieval evaluation |
| `text_embeddings.py` | Qwen document/query encoding, instruction asymmetry, normalization, cache keys | captions, projection training |
| `cache.py` | released-embedding ingestion, versioned object-keyed reads/writes, completeness checks | model policy, scientific selection |
| `reference.py` | load pinned released config/safetensors into `model.py`, state mapping, output-equivalence fixture | training, metric definitions |
| `model.py` | image projector, text projector, normalization, log-parameterized temperature | data loading, checkpoint decisions |
| `losses.py` | pure symmetric InfoNCE computation | optimization, file IO |
| `training.py` | seeding, optimizer/scheduler, epochs, validation loop and Recall@10 checkpoint policy via `metrics.py` | metric formulas, headline benchmark tuning |
| `retrieval.py` | query projection, candidate ranking, top-k row outputs | metric definitions, model training |
| `metrics.py` | sole graded nDCG and Recall@k implementations, aggregation, uncertainty helpers | model execution, dataset discovery |
| `evaluate.py` | frozen benchmark orchestration and result tables | checkpoint or prompt selection |
| `artifacts.py` | run directory lifecycle, atomic metadata writes, completeness checks | scientific calculations |
| `smoke.py` | shared multi-condition caption/embedding-cache orchestration and common-set reduction for the smoke and pilot scripts | metric formulas, split derivation |
| `launch_contract.py` | prerequisite-gate assembly and validation for launch contracts | run execution, metric policy |
| `cli.py` | thin subcommands that connect modules | duplicate pipeline logic |
| `utils.py` | small serialization, filesystem, and hash helpers (JSONL read/append, file hashing) | scientific calculations, config policy |

If a module starts owning two unrelated scientific responsibilities, split it. Do not create interfaces merely because they might be useful later.

## Data pipeline

```text
released datasets + benchmark coordinates
  -> manifest.build
  -> manifest.parquet
     object identity + exclusions + split + source versions
  -> exact train/validation/common-set manifests

released summaries + released OpenAI summary embeddings
  -> cache.ingest_released
  -> released_summary_openai_embeddings.parquet

images selected by manifest
  -> captioning.generate
  -> captions.parquet
  -> text_embeddings.encode_documents
  -> text_embeddings.parquet

released summaries
  -> text_embeddings.encode_documents
  -> released_summary_qwen_embeddings.parquet
```

`manifest.py` is the only place that defines membership or splits. Downstream commands select rows from a manifest but never recompute identity, exclusions, or split assignment.

Every cache row contains `object_id` and provenance sufficient to reject stale reuse: model ID and revision, prompt or instruction hash, preprocessing version, output dimension, source checksum, and a checksum of the float32 vector payload. Cache completion is checked against the input manifest fingerprint. Derived caches are new artifacts whose metadata names the source fingerprint and transform; released artifacts are never mutated in place.

Embedding-cache validation remeasures every vector norm under the config's normalization policy and rejects both out-of-tolerance vectors and stored flags that contradict measurement. Qwen normalization occurs after conversion to float32. Released OpenAI vectors remain verbatim and pass the same `atol=1e-3` policy without renormalization.

Manifest construction asserts that exclusion IDs have empty intersection with train or validation IDs. Document-cache validation rejects any row whose provenance records a non-empty query instruction; these invariants have direct tests.

At real-manifest construction in Phase 2, the run seed must be passed from config rather than a CLI default. Every caption-screen and retrieval-benchmark exclusion must either match at least one source-manifest row or appear in a named absent-object artifact; silent zero-match exclusions are invalid.

The caption audit is a separate bounded pipeline:

```text
same 64 images + same paper free-form prompt
  -> Qwen3-VL-8B descriptions
  -> GPT-4.1-mini descriptions
  -> same text-only Gemma 4 26B A4B judge
       released prompt + XGrammar-enforced released response schema
  -> caption_audit.py
  -> released path-overlap score + secondary per-question audit
```

The closed reference is `openai/gpt-4.1-mini-2025-04-14`, called locally through OpenRouter with the OpenAI provider pinned and no fallback. GPT and Qwen receive the same 64 images and the same free-form prompt from the released paper pipeline. `google/gemma-4-26B-A4B-it` at revision `5305c1e...` then receives description text only. It receives the released GalaxyBench judge prompt verbatim. The released `GalaxyDecisionTree` Pydantic class is converted to JSON Schema, and XGrammar masks invalid next tokens during generation. A small synthetic mapping set must reach 100% answer accuracy before the 64-object judgment starts.

The prompt's request to stay under 300 words is soft verbosity guidance, matching the released pipeline. Caption code validates only that a response is non-empty: it never rejects, retries, or truncates text based on length. Word counts remain descriptive diagnostics. The v2 frozen caption artifacts restore every full response from the immutable raw text captured in v1, so neither captioner is called again.

Gemma calibration and benchmark judging persist raw generated text before parsing. Schema enforcement happens during decoding, not through post-hoc repair, and Pydantic validates the resulting JSON again. The primary score reproduces the released intersection of judge and human-volunteer paths divided by human-path length; per-question accuracy and abstention remain secondary diagnostics.

## Training pipeline

```text
resolved YAML config + condition manifest
  -> config.validate
  -> training.seed before model construction
  -> datasets.make_loaders
  -> model.AIONSearchModel
  -> losses.symmetric_infonce
  -> optimizer step
  -> validation embeddings -> metrics.recall_at_k
  -> best checkpoint + row/batch/step diagnostics
  -> artifacts.finalize
```

The three conditions differ only through config-selected text source and text encoder. They share loaders, model, loss, optimizer logic, validation, and artifact writing.

The first implementation matches the released mean-embedding design: 768-dimensional frozen AION image vectors; residual MLP image and text projectors; normalized 1024-dimensional outputs; symmetric InfoNCE; and a log-parameterized temperature initialized at `log(1/0.07)` whose exponentiated scale is clamped at 100. Architecture experiments wait until this baseline is complete and diagnosed.

## Model contract

- `reference.py` reproduces the packaged `from_pretrained()` route: load pinned `config.json` and `model.safetensors`, construct `model.py`, then load the state dictionary. `image_input_dim`, `text_input_dim`, `embedding_dim`, `image_hidden_dim`, `text_hidden_dim`, `dropout`, and `use_mean_embeddings` must exist; do not fall back to code defaults.
- This project uses mean AION embeddings only. `use_mean_embeddings` must be true and `SimpleImageProjector` is the sole image projector; `CrossAttentionImageProjector` is outside scope.
- The image projector L2-normalizes its input with `eps=1e-6`; the text projector does not normalize its input. Both normalize their outputs with `eps=1e-3`.
- Retrained cells use the released initialization family: Xavier-uniform linear weights, zero linear biases, and dropout 0.1, all resolved through config.
- The equivalence test runs both implementations in eval mode on the same CPU fp32, deliberately non-normalized fixed inputs. Image features, text features, logits, and logit scale must agree with `rtol=0` and `atol=1e-6` before released-checkpoint evaluation is accepted.

## Evaluation pipeline

```text
released config/safetensors -> reference.load_released -> model.py
reimplementation checkpoint -------------------------> model.py

selected model + frozen benchmark + locked query file
  -> condition-specific query embedding
  -> retrieval.rank
  -> ranked_rows.parquet
  -> metrics.grade
  -> metrics.json + tables.csv
```

`reference.py` must match the authors' projected outputs on fixed input tensors before released-checkpoint scores are accepted. One ranking path then serves released and reimplemented models across spiral, merger, and lens evaluation. `metrics.py` contains the sole graded nDCG implementation. Headline evaluation consumes a locked checkpoint; it cannot return a checkpoint-selection signal to training.

Canonical paper queries and preregistered paraphrases are separate outputs. Before results are read, the complete R-OAI query set is embedded once with `text-embedding-3-large` and preserved with response metadata; open conditions embed the same strings locally with the frozen query instruction.

The only local closed-API operations are the one-time R-OAI query freeze and the bounded 64-image GPT-4.1-mini free-form reference. Both use `OPEN_ROUTER_KEY` from the private research `.env`, pin the OpenAI provider without fallback, and write no secret to artifacts. The GPT run records per-object usage and enforces a hard $0.10 budget. Only frozen GPT descriptions move to THQL; the key never does. The released checkpoint gate reproduces the paper's full-set canonical-query AION-Search row: spiral 0.941, merger 0.554, and lens 0.173 nDCG@10, plus exactly two confirmed lenses in the top 10; the re-ranked row is out of scope.

## Config boundary

One YAML file describes one run. It must resolve and save:

- condition and phase;
- dataset, manifest, cache, and benchmark fingerprints;
- model IDs and exact revisions;
- caption prompt/preprocessing/decoding;
- document and query instruction policy;
- projector, loss, and temperature parameterization, initialization, and maximum scale;
- required released-model config keys and `use_mean_embeddings: true`;
- seed, rows, batch size, epochs, optimizer, scheduler;
- validation metric and checkpoint rule;
- query-set revision and output directory.

Unknown keys fail. Destructive reuse of an existing run directory requires an explicit `--overwrite`; the normal response is a new run ID. Scientific constants belong in config or a frozen query/prompt file, not scattered through source code.

## Artifact contract

Each run writes `results/<run_id>/`:

| Artifact | Meaning |
|---|---|
| `config.yaml` | fully resolved experiment definition |
| `command.txt` | exact invocation |
| `run_status.json` | lifecycle, host, device, timestamps, completion |
| `manifest.json` | input fingerprints, row counts, exclusions, split identity |
| `training_history.csv` | epoch loss, validation metrics, learning rate, steps |
| `checkpoint/` | selected projection weights and model metadata |
| `ranked_rows.parquet` | per-query candidates, ranks, scores, relevance |
| `metrics.json` | aggregate metrics derived from ranked rows |
| `tables.csv` | compact report-ready results |
| `errors.jsonl` | structured failures or skipped rows, if any |
| `predictions.parquet` | Phase 6 `spec_probes` only: row-level probe predictions per encoder/target/probe-family/split-seed; primary evidence, `metrics.json` must be recomputable from it |

Caption and text-embedding generation are dataset-cache jobs rather than training runs. They follow the same resolved-config, input-fingerprint, status, error-log, and completeness conventions. Phase 1 writes both free-form description caches, schema-constrained judge outputs, calibration results, released path-overlap audits, secondary per-question audits, and one direct comparison. Both GPT-minus-Qwen deltas use paired object bootstraps and require no additional model call. Local GPT preparation additionally writes image-dimension preflight, per-request usage, and cost artifacts.

The 64-image screen may fail fast on invalid structured output. Before captioning beyond that screen, generation must instead preserve every unparseable response in `errors.jsonl` with object ID and error context so a long cache job remains auditable.

The 1k smoke uses a capped log-and-skip caption path. Failed rows are recorded once, resume treats successes and errors as attempted, and every training condition is reduced to the same successful-object manifest. Its prerequisite contract carries the Phase 0 reference scores and the matched Qwen/GPT Phase 1 diagnostic accuracy/abstention intervals. Only a failed Phase 0 reproducibility gate blocks this engineering smoke; caption-screen performance is evidence to interpret, not a manual launch veto.

Comparisons are valid only when common-set, split, benchmark, and query fingerprints match. Headline values must be regenerable from `ranked_rows.parquet`; `metrics.json` alone is not primary evidence.

## Compute boundary

- Local machine: tests, tiny CPU fixtures, manifest inspection, config validation, pulled-result analysis, the one-time query freeze, and the bounded 64-image GPT reference.
- Laptop safety is a hard boundary: never load or transform full embedding caches locally and never run full vector-payload checksum passes locally. The eGPU may be used without separate approval for bounded GPU work, but it does not relax host RAM/CPU limits. Large cache derivation and manifest-scale data work belongs on THQL after explicit run approval.
- `a100_thql`: primary caption generation, embedding, and training under `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/AION-Search` using the uv environment `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/envs/astrobridge` (Python 3.11, PyTorch 2.5.1+cu121).
- Gemma weights live outside the checkout at `/data2/cmdir/home/ioit_thql/trung_ng/astrobridge/models/gemma-4-26B-A4B-it`. The local staging copy is transfer-only: it is never instantiated on the laptop and may be removed after the remote copy is verified.
- Reuse `/data2/cmdir/home/ioit_thql/.cache/huggingface`; the pinned Qwen3-VL-8B-Instruct snapshot is already complete there. Do not create a project-local duplicate cache.
- `a100_fusion` (`ioit_111`): backup only when THQL is occupied or an approved parallel run is needed.
- Cluster scripts submit through the existing `run_via_slurm` and `script` pattern. Resource lines stay unchanged; experiment choice comes from the entrypoint and config.
- Sync helpers transfer explicit changed files and explicit result paths, never the entire repository blindly.

No GPU job or model/data download starts without approval and an ETA. Every launch report records synced files, job ID, node/GPU, state, log path, output path, and the question the run answers.

When the user asks to stop for review, implementation ends after local code and tests but before GitHub push, cluster sync, or cluster execution. A local closed-API audit or query freeze still requires explicit approval even though it uses neither GPU nor shared cluster data.

## Spectrum encoder probes

Bounded review instrument, encoder-quality half only (decision 12): before any spectrum encoder enters the retrieval/training pipeline, `src/spec_probes/` benchmarks how well its frozen embeddings support recovery of physical labels. Redshift (spec-z) is live; DESI spectral class (SPECTYPE) is not a column the streamed `MultimodalUniverse/desi` sample carries at all, so that probe stays behind `labels.spectral_class.enabled: false` in config until a separate, not-yet-scoped DESI VAC crossmatch deliverable supplies it. The companion cross-match feasibility check (captioned objects x spectra, the wiki plan's Optional Phase 6 week-gate item 3) is the other half of the review and is delivered separately from this package. `src/spec_probes/` is a separate package from `src/aion_reimp/`, with its own config schema and zero import coupling to `aion_reimp.model` / `.training` / `.retrieval` / `.losses` / `.datasets`. It reuses `aion_reimp.utils`, `aion_reimp.artifacts`, and the generic object-level split/fingerprint helpers in `aion_reimp.manifest` (`split_fraction`, `manifest_fingerprint`) rather than re-deriving equivalent logic.

| Module | Owns | Must not own |
|---|---|---|
| `spec_probes/config.py` | strict `phase6_probes` schema, unknown-key rejection | probe fitting, metric formulas |
| `spec_probes/spectra_data.py` | probe sample selection, label extraction, spectrum-batch construction, object-level split + fingerprint | encoder-specific code, probe fitting |
| `spec_probes/encoders.py` | three frozen encoder adapters behind one shared interface; model-specific code; device/batch/dtype policy | split/label logic, metric formulas |
| `spec_probes/specformer_model.py` | vendored, frozen-inference-only SpecFormer architecture (source-cited) | training, encoder-selection logic |
| `spec_probes/probes.py` | ridge/logistic/kNN fitting and prediction; fold-local standardization during CV; outer-train standardization for final fits; trivial baselines (pure) | metric formulas, file IO |
| `spec_probes/probe_metrics.py` | NMAD, catastrophic-outlier fraction, MAE, R2, accuracy, macro-F1 (pure) | probe fitting, encoder code |
| `spec_probes/run_probes.py` | per-encoder, per-split-seed probe orchestration, row-level predictions, metrics recomputed from predictions, seed aggregation | encoder-specific code, retrieval/training |

A `--preflight` mode of `scripts/run_phase6_probes_cluster.py` streams and reports the first record's schema, repeats the seeded stream selection and requires identical ordered object IDs, loads both neural checkpoints, and encodes enough real rows to cover both the configured neural batch sizes and PCA component count. It requires the requested device, reports free GPU memory, and writes one report to the path named by `run.preflight_report` without creating a `results/<run_id>` directory. The report is bound to the exact resolved config, relevant code-file hashes, package versions, Python version, and device. `cluster/script_phase6_probes.sh` runs only this preflight; after manual review, `cluster/script_phase6_probes_full.sh` starts the full run. The full run refuses to create its run directory unless the bound report still matches and reports `status: "pass"`.

```text
MMU DESI streaming sample (ZWARN "no problem", probe-scale)
  -> spectra_data.select_probe_sample / extract_labels / extract_spectrum_batch
  -> per split.seeds entry: spectra_data.object_level_split (reuses
     aion_reimp.manifest.split_fraction)
     one seeded train/test split, fingerprinted, shared by every encoder
     and both probe families within that split seed -- no per-encoder split drift

frozen neural encoder (AION-1 | AstroCLIP SpecFormer)
  -> build/load once -> embed the full selected 10k sample once (chunked,
     config device/batch/dtype) -> reuse row-aligned CPU embeddings across
     every split seed; no repeated frozen inference caused by split choice

PCA baseline
  -> fit and embed separately inside each split seed because its basis is
     outer-train-fitted and therefore split-dependent

each encoder/split -> test-embedding fingerprint + reuse provenance
  -> probes.make_cv_folds (shared across encoders) + fold-local scaling for
     alpha/C selection; final scaler fit on the complete outer train split
  -> probes.select_ridge_alpha / select_logistic_c -> ridge_probe / logistic_probe /
     knn_regression_probe / knn_classification_probe, plus probes.median_baseline_predict /
     majority_class_baseline_predict (encoder-independent)
  -> run_probes.run_probe_suite_for_encoder / run_baseline_suite
  -> predictions.parquet (row-level, one split_seed column per row, primary evidence)
  -> run_probes.metrics_from_predictions (per split seed) ->
     run_probes.aggregate_seed_metrics (mean/std across split.seeds) -> metrics.json + tables.csv
```

## Spectrum crossmatch feasibility probe

The other half of decision 12 is owned by `src/spectra_crossmatch/`, not by the encoder probe or retrieval packages. It reuses the exact completed Phase-3 caption manifest and its saved coordinates; it must not resample `astronolan/galaxy-descriptions`. The right side is the pinned `UniverseTBD/mmu_desi_edr_sv3` HATS catalog, opened through LSDB with only `object_id`, RA/Dec, `Z`, `ZERR`, and `ZWARN`. Spectrum arrays are never loaded for the feasibility count.

| Module | Owns | Must not own |
|---|---|---|
| `spectra_crossmatch/config.py` | strict `phase6_crossmatch` schema and cross-field policy | network access, matching, output writing |
| `spectra_crossmatch/crossmatch.py` | exact source binding, stable LSDB output normalization, spectrum-quality flags, duplicate ranking, radius/survey summaries | LSDB/HF calls, model code, retrieval design |
| `scripts/run_phase6_crossmatch_cluster.py` | bound preflight, column-pruned LSDB execution, run artifacts | metric invention, model inference, caption generation |

```text
Phase-3 common_manifest.parquet + data/source_rows.parquet
  -> exact 10k captioned object IDs + survey + RA/Dec (fingerprinted)
  -> LSDB in-memory HATS left catalog

pinned MMU DESI EDR HATS catalog (metadata columns only, 10-arcsec margin)
  -> one 3-arcsec crossmatch, up to 8 candidates per caption object
  -> candidate_matches.parquet (all candidates; fail if the cap is reached)
        -> quality flag: HATS raw ZWARN False + finite Z >= 0 + finite ZERR > 0
  -> deterministic duplicate choice: separation, then DESI object_id
  -> selected_matches.parquet + counts_by_radius.csv (0.5/1/2/3 arcsec)
     + counts_by_survey.csv + summary.json
```

`--preflight` binds the exact local Phase-3 artifacts, current config/code/package versions, pinned Hub revision, HATS schema, right-margin presence, and one real zero-distance DESI self-match. It writes no run directory. The full CPU-only job refuses to start from a failed or stale report.

## Decisions that remain stable

1. `orig_repo/` is reference-only; no new implementation is added inside it.
2. Configs choose conditions; source files do not contain condition-specific branches beyond validated adapters.
3. Manifest fingerprints define experimental populations and splits.
4. Benchmark overlap is removed before splitting.
5. Seeds are set before model and projection-head creation.
6. Pilot loaders must expose at least four optimizer steps per epoch.
7. Validation Recall@10 selects checkpoints; final Galaxy Zoo and lens metrics do not.
8. Document embeddings have no instruction; a document cache carrying a query instruction fails validation.
9. Row-level rankings are primary evaluation evidence.
10. Existing result directories are never silently mixed with retries.
11. A new model arm requires a specific unresolved scientific question.
12. Spectra code is not added to the retrieval/training pipeline until cross-match size and a spectra-sensitive evaluation pass review. That review has two halves, delivered separately: the frozen-encoder physical-recovery probes in `src/spec_probes/` (see "Spectrum encoder probes" below) are the encoder-quality half, and a captioned-objects x spectra cross-match feasibility check (the wiki plan's Optional Phase 6 week-gate item 3) is the other half. `src/spec_probes/` has no retrieval or training coupling.

## Change checklist

Before accepting a code change:

- Does the config still explain the complete run?
- Is every new config key consumed or rejected?
- Did split or exclusion logic leak outside `manifest.py`?
- Did model-specific instruction logic leak outside caption/text adapters?
- Can cached outputs be matched one-to-one to manifest object IDs?
- Can metrics be recomputed from row-level rankings?
- Does a retry require an explicit new run ID or overwrite flag?
- Is this needed by the current experiment, or is it premature engineering?
- If a module, pipeline edge, config field, or artifact changed, was this document updated?
