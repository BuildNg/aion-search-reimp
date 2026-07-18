"""Phase 2/3 smoke-scale data preparation: source selection, audited exclusions,
Qwen captioning, and the common embedding-cache set they train against."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from .cache import (
    NormalizationPolicy,
    cache_fingerprint,
    ingest_released_embeddings,
    load_cache_subset,
    validate_embedding_cache,
    write_embedding_cache,
)
from .captioning import QwenCaptioner, append_caption_results
from .datasets import _to_image, load_pinned_dataset
from .manifest import (
    assert_exclusion_coverage,
    assert_no_benchmark_leakage,
    build_manifest,
    coordinate_exclusion_coverage,
    exact_exclusion_coverage,
    manifest_fingerprint,
    write_manifest,
)
from .text_embeddings import EmbeddingSpec, QwenEmbedder, embedding_frame


def _selection_key(object_id: str, seed: int) -> str:
    return hashlib.sha256(f"smoke|{object_id}|{seed}".encode("utf-8")).hexdigest()


def select_smoke_rows(manifest: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    assert_no_benchmark_leakage(manifest)
    eligible = manifest[manifest["split"].isin(["train", "validation"])].copy()
    if len(eligible) < sample_size:
        raise ValueError(f"Only {len(eligible)} eligible rows for sample_size={sample_size}")
    eligible["selection_key"] = eligible["object_id"].map(
        lambda value: _selection_key(str(value), seed)
    )
    selected = eligible.sort_values(["selection_key", "object_id"]).head(sample_size).copy()
    selected["selection_rank"] = np.arange(len(selected), dtype=np.int64)
    selected = selected.drop(columns="selection_key")
    if set(selected["split"]) != {"train", "validation"}:
        raise AssertionError("Smoke sample must contain train and validation rows")
    return selected.reset_index(drop=True)


def prepare_smoke_source(
    source_spec: Mapping[str, Any],
    exclusion_spec: Mapping[str, Any],
    output_dir: Path,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize one common image/vector set and complete exclusion coverage."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Smoke data output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_pinned_dataset(
        source_spec["repo_id"], source_spec["revision"], source_spec["split"]
    )
    source_columns = {
        source_spec["object_id_column"],
        source_spec["survey_column"],
        source_spec["ra_column"],
        source_spec["dec_column"],
        source_spec["image_column"],
        source_spec["image_embedding_column"],
        source_spec["released_text_column"],
        source_spec["released_embedding_column"],
    }
    missing = source_columns - set(dataset.column_names)
    if missing:
        raise ValueError(f"Smoke source dataset missing columns: {sorted(missing)}")

    metadata_columns = [
        source_spec["object_id_column"],
        source_spec["survey_column"],
        source_spec["ra_column"],
        source_spec["dec_column"],
    ]
    metadata = dataset.select_columns(metadata_columns).to_pandas()
    metadata = metadata.rename(
        columns={
            source_spec["object_id_column"]: "object_id",
            source_spec["survey_column"]: "survey",
            source_spec["ra_column"]: "ra",
            source_spec["dec_column"]: "dec",
        }
    )
    metadata["object_id"] = metadata["object_id"].astype(str)
    metadata["source_row_id"] = np.arange(len(metadata), dtype=np.int64)

    screen = pd.read_parquet(Path(exclusion_spec["caption_screen_labels"]))
    if "object_id" not in screen.columns:
        raise ValueError("Caption-screen exclusion artifact requires object_id")
    exact_coverage = exact_exclusion_coverage(
        metadata["object_id"], "caption_screen_64", screen["object_id"].astype(str)
    )
    benchmark_frames = {
        name: pd.read_parquet(Path(path))
        for name, path in exclusion_spec["benchmark_coordinates"].items()
    }
    coordinate_coverage = coordinate_exclusion_coverage(
        metadata,
        benchmark_frames,
        radius_arcsec=float(exclusion_spec["radius_arcsec"]),
    )
    coverage = pd.concat([exact_coverage, coordinate_coverage], ignore_index=True)
    expected_coverage_rows = len(screen) + sum(len(frame) for frame in benchmark_frames.values())
    assert_exclusion_coverage(coverage, expected_rows=expected_coverage_rows)
    coverage.to_parquet(output_dir / "exclusion_coverage.parquet", index=False)

    matched = coverage.loc[coverage["status"].eq("matched"), "source_object_id"].astype(str)
    manifest = build_manifest(
        metadata,
        {"caption_screen_and_retrieval_benchmarks": matched},
        seed=seed,
        train_ratio=float(source_spec["train_ratio"]),
        image_embedding_version=f"{source_spec['repo_id']}@{source_spec['revision']}",
    )
    selected_manifest = select_smoke_rows(manifest, int(source_spec["sample_size"]), seed)
    write_manifest(selected_manifest, output_dir / "manifest.parquet")

    image_dir = output_dir / "images"
    image_dir.mkdir()
    selected_rows = dataset.select(selected_manifest["source_row_id"].astype(int).tolist())
    manifest_by_id = selected_manifest.set_index("object_id")
    records = []
    for row in selected_rows:
        object_id = str(row[source_spec["object_id_column"]])
        image_name = hashlib.sha256(object_id.encode("utf-8")).hexdigest() + ".png"
        image_path = image_dir / image_name
        _to_image(row[source_spec["image_column"]]).save(image_path)
        manifest_row = manifest_by_id.loc[object_id]
        image_embedding = np.asarray(
            row[source_spec["image_embedding_column"]], dtype=np.float32
        )
        released_embedding = np.asarray(
            row[source_spec["released_embedding_column"]], dtype=np.float32
        )
        if image_embedding.size != 768:
            raise ValueError(f"Expected 768 AION dimensions for object_id={object_id}")
        if released_embedding.size != 3072:
            raise ValueError(f"Expected 3072 OpenAI dimensions for object_id={object_id}")
        records.append(
            {
                "object_id": object_id,
                "survey": str(row[source_spec["survey_column"]]),
                "ra": float(row[source_spec["ra_column"]]),
                "dec": float(row[source_spec["dec_column"]]),
                "source_row_id": int(manifest_row["source_row_id"]),
                "split": str(manifest_row["split"]),
                "image_path": str(image_path.resolve()),
                "image_embedding": image_embedding.tolist(),
                "released_summary": str(row[source_spec["released_text_column"]]),
                "released_openai_embedding": released_embedding.tolist(),
            }
        )
    source_frame = pd.DataFrame(records).sort_values("object_id").reset_index(drop=True)
    if set(source_frame["object_id"]) != set(selected_manifest["object_id"]):
        raise AssertionError("Materialized source rows do not match the smoke manifest")
    source_rows_path = output_dir / "source_rows.parquet"
    source_frame.to_parquet(source_rows_path, index=False)
    # Row-order-invariant content fingerprint of every column, including the image
    # and released-text embeddings, not just the object_id set. A seed-extension
    # run that later loads this file (see load_cached_smoke_source) compares
    # against this recorded value, so a change to e.g. an embedding column while
    # object IDs stay fixed is caught rather than silently reused.
    source_rows_path.with_suffix(source_rows_path.suffix + ".meta.json").write_text(
        json.dumps(
            {
                "rows": len(source_frame),
                "content_fingerprint": manifest_fingerprint(source_frame),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "data_summary.json").write_text(
        json.dumps(
            {
                "source_rows": len(source_frame),
                "train_rows": int(source_frame["split"].eq("train").sum()),
                "validation_rows": int(source_frame["split"].eq("validation").sum()),
                "exclusion_rows": len(coverage),
                "matched_exclusions": int(coverage["status"].eq("matched").sum()),
                "absent_exclusions": int(coverage["status"].eq("absent").sum()),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return source_frame, selected_manifest


def generate_qwen_captions_and_common_set(
    caption_spec: Mapping[str, Any],
    source: pd.DataFrame,
    manifest: pd.DataFrame,
    output_root: Path,
    join_mismatch_message: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate Qwen captions, then reduce source/manifest to captioned objects.

    Writes ``q_qwen_captions.jsonl``, ``errors.jsonl``, ``caption_generation.json``,
    and ``common_manifest.parquet`` under ``output_root``. ``join_mismatch_message``
    lets each Phase 2/3 caller keep its own assertion text for the (should-never-
    happen) case where the caption output does not join one-to-one with the source.
    Returns (captions, common_source, common_manifest, caption_stats).
    """
    captioner = QwenCaptioner(
        caption_spec["model_id"],
        caption_spec["revision"],
        Path(caption_spec["prompt_file"]).read_text(encoding="utf-8"),
        dtype=caption_spec["dtype"],
        max_new_tokens=caption_spec["max_new_tokens"],
    )
    captions_path = output_root / "q_qwen_captions.jsonl"
    error_path = output_root / "errors.jsonl"
    caption_stats = append_caption_results(
        captioner,
        source.loc[:, ["object_id", "image_path"]].to_dict(orient="records"),
        captions_path,
        error_jsonl=error_path,
        max_error_rate=float(caption_spec["max_error_rate"]),
    )
    (output_root / "caption_generation.json").write_text(
        json.dumps(caption_stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    del captioner
    gc.collect()
    torch.cuda.empty_cache()

    captions = pd.read_json(captions_path, lines=True)
    successful_ids = set(captions["object_id"].astype(str))
    common_source = source[source["object_id"].astype(str).isin(successful_ids)].copy()
    common_manifest = manifest[
        manifest["object_id"].astype(str).isin(successful_ids)
    ].copy()
    if len(common_source) != len(captions):
        raise AssertionError(join_mismatch_message)
    write_manifest(common_manifest, output_root / "common_manifest.parquet")
    return captions, common_source, common_manifest, caption_stats


def build_common_text_embedding_caches(
    common_source: pd.DataFrame,
    captions: pd.DataFrame,
    text_embedding_spec: Mapping[str, Any],
    caches_spec: Mapping[str, Any],
    caption_prompt_file: str,
    output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, "EmbeddingSpec", "QwenEmbedder"]:
    """Build the released-OpenAI, released-Qwen, and Qwen-caption embedding caches.

    Writes ``r_oai_embeddings.parquet``, ``r_qwen_embeddings.parquet``, and
    ``q_qwen_embeddings.parquet`` (plus ``.meta.json`` siblings) under
    ``output_root``. Returns the embedder still constructed (not deleted) so a
    caller that also embeds a query set, such as Phase 3, can reuse it before
    releasing GPU memory itself.
    """
    policy = NormalizationPolicy(
        required=True, atol=float(text_embedding_spec["normalization_atol"])
    )
    r_oai = ingest_released_embeddings(
        common_source.rename(
            columns={
                "released_summary": "summary",
                "released_openai_embedding": "summary_text_embedding",
            }
        ),
        text_column="summary",
        embedding_column="summary_text_embedding",
        normalization_policy=policy,
    )
    write_embedding_cache(
        r_oai,
        output_root / "r_oai_embeddings.parquet",
        normalization_policy=policy,
        metadata={"transform": "released_verbatim_subset"},
    )

    r_qwen_source_path = (
        Path(caches_spec["phase1_normalized_dir"]) / caches_spec["released_summary_qwen"]
    )
    r_qwen = load_cache_subset(r_qwen_source_path, common_source["object_id"])
    validate_embedding_cache(r_qwen, policy)
    r_qwen_source_meta = json.loads(
        r_qwen_source_path.with_suffix(
            r_qwen_source_path.suffix + ".meta.json"
        ).read_text(encoding="utf-8")
    )
    write_embedding_cache(
        r_qwen,
        output_root / "r_qwen_embeddings.parquet",
        normalization_policy=policy,
        metadata={
            "source_path": str(r_qwen_source_path),
            "source_fingerprint": r_qwen_source_meta["fingerprint"],
            "subset_fingerprint": cache_fingerprint(r_qwen),
            "transform": "manifest_subset",
        },
    )

    embedding_spec = EmbeddingSpec.from_mapping(text_embedding_spec)
    embedder = QwenEmbedder(embedding_spec)
    q_vectors = embedder.encode(captions["description"].astype(str).tolist(), "document")
    q_qwen = embedding_frame(
        captions["object_id"].astype(str).tolist(),
        captions["description"].astype(str).tolist(),
        q_vectors,
        "document",
        embedding_spec,
    )
    write_embedding_cache(
        q_qwen,
        output_root / "q_qwen_embeddings.parquet",
        normalization_policy=embedding_spec.normalization_policy,
        metadata={"caption_prompt": caption_prompt_file},
    )
    return r_oai, r_qwen, q_qwen, embedding_spec, embedder


def load_cached_smoke_source(
    base_output_root: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load a previously materialized Phase 3 source/manifest pair from a prior run.

    A seed-extension run must select the identical 10k manifest a prior run already
    built (same ``run.seed``, same source/exclusion config) without re-downloading
    images or recomputing the split. This reads ``data/source_rows.parquet`` and
    ``data/manifest.parquet`` under ``base_output_root`` and fails loudly (rather than
    falling back to materializing a fresh source) if either file is missing or its
    fingerprint does not match its recorded metadata.

    Cache identity policy for ``source_rows.parquet``: the object-ID-set check above
    only proves the same rows are present, not that their contents (e.g. image
    embeddings) are unchanged. This loader also recomputes a row-order-invariant
    content fingerprint of the loaded source rows -- reusing
    :func:`aion_reimp.manifest.manifest_fingerprint`, which sorts by ``object_id``
    before hashing every column, so it covers embedding columns as well as the ID
    set -- and compares it against ``source_rows.parquet.meta.json``'s
    ``content_fingerprint`` field. Runs created by :func:`prepare_smoke_source` after
    this check was added write that field, and a mismatch there is fatal. A base run
    written before this check existed (for example ``phase3_10k_v1``) has no
    recorded content fingerprint to compare against; in that documented fallback
    case, this loader computes the fingerprint once and records its status as
    ``"source_content_fingerprint_computed_not_verified"`` in the returned
    provenance mapping instead of raising, because there is nothing on disk to
    verify it against. Callers (see ``validate_base_run``) are expected to log that
    provenance mapping prominently rather than treat it as a passed check.

    Returns ``(source, manifest, source_provenance)``, where ``source_provenance``
    has keys ``manifest_fingerprint``, ``source_content_fingerprint``, and
    ``source_content_fingerprint_status``.
    """
    base_output_root = Path(base_output_root)
    manifest_path = base_output_root / "data" / "manifest.parquet"
    source_path = base_output_root / "data" / "source_rows.parquet"
    if not manifest_path.exists() or not source_path.exists():
        raise FileNotFoundError(
            "Cache-only mode requires an existing smoke source at "
            f"{base_output_root / 'data'}; missing "
            f"{manifest_path if not manifest_path.exists() else source_path}"
        )
    manifest = pd.read_parquet(manifest_path)
    manifest_meta = json.loads(
        manifest_path.with_suffix(manifest_path.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest_fingerprint(manifest) != manifest_meta["fingerprint"]:
        raise ValueError(
            f"Cached manifest at {manifest_path} does not match its recorded "
            "fingerprint (cache miss)"
        )
    source = pd.read_parquet(source_path)
    if set(source["object_id"].astype(str)) != set(manifest["object_id"].astype(str)):
        raise ValueError(
            "Cached source rows and cached manifest disagree on object IDs (cache miss)"
        )

    source_content_fingerprint = manifest_fingerprint(source)
    source_meta_path = source_path.with_suffix(source_path.suffix + ".meta.json")
    recorded_content_fingerprint = None
    if source_meta_path.exists():
        recorded_content_fingerprint = json.loads(
            source_meta_path.read_text(encoding="utf-8")
        ).get("content_fingerprint")
    if recorded_content_fingerprint is not None:
        if source_content_fingerprint != recorded_content_fingerprint:
            raise ValueError(
                f"Cached source rows at {source_path} do not match their recorded "
                "content fingerprint (cache miss): source_rows.parquet content "
                "changed while object IDs stayed the same"
            )
        source_content_fingerprint_status = "verified_against_recorded_fingerprint"
    else:
        source_content_fingerprint_status = "source_content_fingerprint_computed_not_verified"

    source_provenance = {
        "manifest_fingerprint": manifest_meta["fingerprint"],
        "source_content_fingerprint": source_content_fingerprint,
        "source_content_fingerprint_status": source_content_fingerprint_status,
    }
    return source, manifest, source_provenance


def load_cached_common_set(
    base_output_root: Path,
    source: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Load a previously generated Qwen caption set instead of calling the captioner.

    Mirrors the return shape of :func:`generate_qwen_captions_and_common_set` but never
    constructs a ``QwenCaptioner`` or runs generation. Fails loudly on any missing file,
    fingerprint mismatch, or object-ID disagreement between the cached captions and the
    cached common manifest (cache miss) instead of silently falling back to captioning.
    """
    base_output_root = Path(base_output_root)
    captions_path = base_output_root / "q_qwen_captions.jsonl"
    common_manifest_path = base_output_root / "common_manifest.parquet"
    if not captions_path.exists() or not common_manifest_path.exists():
        raise FileNotFoundError(
            f"Cache-only mode requires existing captions at {captions_path} and an "
            f"existing common manifest at {common_manifest_path}"
        )
    captions = pd.read_json(captions_path, lines=True)
    common_manifest = pd.read_parquet(common_manifest_path)
    common_manifest_meta = json.loads(
        common_manifest_path.with_suffix(common_manifest_path.suffix + ".meta.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest_fingerprint(common_manifest) != common_manifest_meta["fingerprint"]:
        raise ValueError(
            f"Cached common manifest at {common_manifest_path} does not match its "
            "recorded fingerprint (cache miss)"
        )
    successful_ids = set(captions["object_id"].astype(str))
    if successful_ids != set(common_manifest["object_id"].astype(str)):
        raise ValueError(
            "Cached captions and cached common manifest disagree on object IDs (cache miss)"
        )
    common_source = source[source["object_id"].astype(str).isin(successful_ids)].copy()
    if len(common_source) != len(captions):
        raise ValueError(
            "Cached common source and cached captions do not join one-to-one (cache miss)"
        )
    caption_stats_path = base_output_root / "caption_generation.json"
    caption_stats: Dict[str, Any] = (
        json.loads(caption_stats_path.read_text(encoding="utf-8"))
        if caption_stats_path.exists()
        else {}
    )
    caption_stats = {**caption_stats, "cache_reused_from": str(base_output_root)}
    return captions, common_source, common_manifest, caption_stats


def load_cached_text_embedding_caches(
    base_output_root: Path,
    text_embedding_spec: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, "EmbeddingSpec"]:
    """Load the released-OpenAI, released-Qwen, and Qwen-caption embedding caches a
    prior Phase 3 run already wrote, instead of re-embedding text.

    Fails loudly on a missing file or a fingerprint mismatch against the recorded
    ``.meta.json`` (cache miss) rather than silently recomputing embeddings.
    """
    base_output_root = Path(base_output_root)
    embedding_spec = EmbeddingSpec.from_mapping(text_embedding_spec)
    document_policy = NormalizationPolicy(
        required=True, atol=float(text_embedding_spec["normalization_atol"])
    )

    def _load(name: str, policy: NormalizationPolicy) -> pd.DataFrame:
        path = base_output_root / f"{name}_embeddings.parquet"
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Cache-only mode requires an existing embedding cache at {path}"
            )
        frame = pd.read_parquet(path)
        validate_embedding_cache(frame, policy)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cache_fingerprint(frame) != meta["fingerprint"]:
            raise ValueError(
                f"Cached embedding table at {path} does not match its recorded "
                "fingerprint (cache miss)"
            )
        return frame

    r_oai = _load("r_oai", document_policy)
    r_qwen = _load("r_qwen", document_policy)
    q_qwen = _load("q_qwen", embedding_spec.normalization_policy)
    return r_oai, r_qwen, q_qwen, embedding_spec


# Config sections compared byte-for-byte between a seed-extension run and the base
# run it reuses caches from. Everything that defines data identity or training
# behavior is named here explicitly (rather than diffing the whole config dict) so
# a mismatch names the divergent key instead of just saying "configs differ".
# run.id, the top-level seeds list, source_run, and kind are the only config
# elements allowed to differ, since those are exactly what make a seed-extension
# run an extension of the base run rather than a duplicate of it. run.seed (the
# manifest-selection seed) is compared separately below, because it lives inside
# the run section alongside the excepted run.id.
COMPARED_BASE_RUN_CONFIG_SECTIONS: Tuple[str, ...] = (
    "queries",
    "prerequisites",
    "source_data",
    "exclusions",
    "captioning",
    "text_embedding",
    "caches",
    "model",
    "training",
    "conditions",
    "benchmarks",
)


def _first_divergence(path: str, base_value: Any, extension_value: Any) -> Optional[Tuple[str, Any, Any]]:
    """Return ``(path, base_leaf, extension_leaf)`` for the first differing leaf, or
    ``None`` if the two values are equal. Recurses into dicts (by key) and
    equal-length lists (by index) so the returned path names the divergent key
    directly, e.g. ``"source_data.sample_size"`` or ``"conditions[1].name"``.
    """
    if base_value == extension_value:
        return None
    if isinstance(base_value, dict) and isinstance(extension_value, dict):
        for key in sorted(set(base_value) | set(extension_value)):
            child = _first_divergence(
                f"{path}.{key}", base_value.get(key, "<missing>"), extension_value.get(key, "<missing>")
            )
            if child is not None:
                return child
        return (path, base_value, extension_value)
    if (
        isinstance(base_value, list)
        and isinstance(extension_value, list)
        and len(base_value) == len(extension_value)
    ):
        for index, (base_item, extension_item) in enumerate(zip(base_value, extension_value)):
            child = _first_divergence(f"{path}[{index}]", base_item, extension_item)
            if child is not None:
                return child
        return (path, base_value, extension_value)
    return (path, base_value, extension_value)


def _assert_section_equal(section_name: str, base_value: Any, extension_value: Any) -> None:
    divergence = _first_divergence(section_name, base_value, extension_value)
    if divergence is not None:
        divergent_path, base_leaf, extension_leaf = divergence
        raise ValueError(
            f"Base run and extension config diverge at {divergent_path}: "
            f"base={base_leaf!r} extension={extension_leaf!r}"
        )


def validate_base_run(
    base_output_root: Path,
    extension_config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail fast, before any cache is loaded, unless the base Phase 3 run this seed
    extension reuses is actually usable.

    Enforces, each fatal on the first violation found:

    - the base run's ``run_status.json`` records ``status == "complete"``;
    - the base run's summary artifact (``phase3_10k_summary.json``) exists and its
      ``"seeds"`` field is exactly ``extension_config["source_run"]["reused_seeds"]``
      -- proof the base run actually finished training and evaluating those seeds,
      not just that its config named them;
    - ``run.seed`` is equal between the base run's saved ``config.yaml`` and the
      extension config (the manifest-selection seed must match for the 10k row
      selection and split to be identical -- previously only a comment-level
      convention, see ``configs/phase3_10k_seedext.yaml``);
    - every section in ``COMPARED_BASE_RUN_CONFIG_SECTIONS`` is equal between the
      base run's saved config and the extension config, naming the first divergent
      key on mismatch;
    - ``source_rows.parquet`` actually corresponds to the recorded
      ``manifest.parquet`` (delegated to :func:`load_cached_smoke_source`, which
      also performs the source-content fingerprint check documented in its own
      docstring).

    Returns a mapping with the loaded ``base_config``, ``run_status``, ``summary``,
    the loaded ``source`` and ``manifest`` frames, the ``source_provenance`` mapping
    from :func:`load_cached_smoke_source`, and the list of ``compared_sections`` --
    so a caller that passes this validation does not have to load any of it twice.
    """
    base_output_root = Path(base_output_root)
    config_path = base_output_root / "config.yaml"
    run_status_path = base_output_root / "run_status.json"
    summary_path = base_output_root / "phase3_10k_summary.json"
    for path in (config_path, run_status_path, summary_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Base run validation requires {path}; the base run at "
                f"{base_output_root} is not usable as a seed-extension source"
            )

    base_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if run_status.get("status") != "complete":
        raise ValueError(
            f"Base run at {base_output_root} has run_status.json status="
            f"{run_status.get('status')!r}, not 'complete'; refusing to extend an "
            "unfinished or failed run"
        )

    reused_seeds = list(extension_config["source_run"]["reused_seeds"])
    base_seeds = summary.get("seeds")
    if base_seeds != reused_seeds:
        raise ValueError(
            f"Base run at {base_output_root} actually ran seeds {base_seeds!r}, but "
            f"source_run.reused_seeds names {reused_seeds!r}"
        )

    base_run_seed = base_config.get("run", {}).get("seed")
    extension_run_seed = extension_config.get("run", {}).get("seed")
    if base_run_seed != extension_run_seed:
        raise ValueError(
            "Base run and extension config diverge at run.seed: "
            f"base={base_run_seed!r} extension={extension_run_seed!r}"
        )

    for section_name in COMPARED_BASE_RUN_CONFIG_SECTIONS:
        _assert_section_equal(
            section_name,
            base_config.get(section_name, "<missing>"),
            extension_config.get(section_name, "<missing>"),
        )

    source, manifest, source_provenance = load_cached_smoke_source(base_output_root)

    return {
        "base_config": base_config,
        "run_status": run_status,
        "summary": summary,
        "source": source,
        "manifest": manifest,
        "source_provenance": source_provenance,
        "compared_sections": ["run.seed", *COMPARED_BASE_RUN_CONFIG_SECTIONS],
    }
