import json

import numpy as np
import pandas as pd
import pytest

from aion_reimp.cache import NormalizationPolicy, ingest_released_embeddings, write_embedding_cache
from aion_reimp.manifest import build_manifest, manifest_fingerprint, write_manifest
from aion_reimp.smoke import (
    load_cached_common_set,
    load_cached_smoke_source,
    load_cached_text_embedding_caches,
)
from aion_reimp.text_embeddings import EmbeddingSpec, embedding_frame


def _manifest_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "object_id": ["a", "b", "c"],
            "survey": ["legacy", "legacy", "hsc"],
            "ra": [1.0, 2.0, 3.0],
            "dec": [-1.0, -2.0, -3.0],
            "source_row_id": [0, 1, 2],
        }
    )


def _write_cached_smoke_source(base_output_root, with_content_fingerprint: bool = True) -> pd.DataFrame:
    manifest = build_manifest(_manifest_source(), {}, seed=7)
    write_manifest(manifest, base_output_root / "data" / "manifest.parquet")
    source = pd.DataFrame(
        {
            "object_id": manifest["object_id"],
            "split": manifest["split"],
            "image_embedding": [[0.1, 0.2]] * len(manifest),
        }
    )
    (base_output_root / "data").mkdir(parents=True, exist_ok=True)
    source_path = base_output_root / "data" / "source_rows.parquet"
    source.to_parquet(source_path, index=False)
    if with_content_fingerprint:
        source_path.with_suffix(source_path.suffix + ".meta.json").write_text(
            json.dumps(
                {"rows": len(source), "content_fingerprint": manifest_fingerprint(source)}
            ),
            encoding="utf-8",
        )
    return source


def test_load_cached_smoke_source_round_trips(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    source = _write_cached_smoke_source(base_output_root)

    loaded_source, loaded_manifest, provenance = load_cached_smoke_source(base_output_root)

    assert set(loaded_source["object_id"]) == set(source["object_id"])
    assert set(loaded_manifest["object_id"]) == {"a", "b", "c"}
    assert provenance["source_content_fingerprint_status"] == "verified_against_recorded_fingerprint"
    assert provenance["source_content_fingerprint"] == manifest_fingerprint(source)


def test_load_cached_smoke_source_missing_file_fails_loudly(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    base_output_root.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="Cache-only mode requires"):
        load_cached_smoke_source(base_output_root)


def test_load_cached_smoke_source_detects_fingerprint_mismatch(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    _write_cached_smoke_source(base_output_root)
    manifest_path = base_output_root / "data" / "manifest.parquet"
    manifest = pd.read_parquet(manifest_path)
    manifest.loc[0, "survey"] = "tampered"
    manifest.to_parquet(manifest_path, index=False)

    with pytest.raises(ValueError, match="cache miss"):
        load_cached_smoke_source(base_output_root)


def test_load_cached_smoke_source_detects_content_fingerprint_mismatch(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    _write_cached_smoke_source(base_output_root)
    source_path = base_output_root / "data" / "source_rows.parquet"
    source = pd.read_parquet(source_path)
    # Object IDs are unchanged; only an embedding value changes. The existing
    # object-ID-set check alone would miss this.
    source.at[0, "image_embedding"] = [9.9, 9.9]
    source.to_parquet(source_path, index=False)

    with pytest.raises(ValueError, match="cache miss"):
        load_cached_smoke_source(base_output_root)


def test_load_cached_smoke_source_missing_content_fingerprint_falls_back(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    source = _write_cached_smoke_source(base_output_root, with_content_fingerprint=False)

    loaded_source, loaded_manifest, provenance = load_cached_smoke_source(base_output_root)

    assert (
        provenance["source_content_fingerprint_status"]
        == "source_content_fingerprint_computed_not_verified"
    )
    assert provenance["source_content_fingerprint"] == manifest_fingerprint(source)


def test_load_cached_common_set_reads_existing_captions(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    source = _write_cached_smoke_source(base_output_root)

    captions = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "description": ["A spiral galaxy.", "A merging pair."],
            "raw_response": ["A spiral galaxy.", "A merging pair."],
            "word_count": [3, 3],
        }
    )
    captions.to_json(base_output_root / "q_qwen_captions.jsonl", orient="records", lines=True)

    common_manifest = build_manifest(_manifest_source(), {}, seed=7)
    common_manifest = common_manifest[common_manifest["object_id"].isin(["a", "b"])]
    write_manifest(common_manifest, base_output_root / "common_manifest.parquet")

    (base_output_root / "caption_generation.json").write_text(
        json.dumps({"completed_rows": 2}), encoding="utf-8"
    )

    loaded_captions, common_source, loaded_common_manifest, caption_stats = load_cached_common_set(
        base_output_root, source
    )

    assert set(loaded_captions["object_id"]) == {"a", "b"}
    assert set(common_source["object_id"]) == {"a", "b"}
    assert set(loaded_common_manifest["object_id"]) == {"a", "b"}
    assert caption_stats["completed_rows"] == 2
    assert caption_stats["cache_reused_from"] == str(base_output_root)


def test_load_cached_common_set_missing_files_fail_loudly(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    source = _write_cached_smoke_source(base_output_root)

    with pytest.raises(FileNotFoundError, match="Cache-only mode requires existing captions"):
        load_cached_common_set(base_output_root, source)


def test_load_cached_common_set_detects_object_id_mismatch(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    source = _write_cached_smoke_source(base_output_root)

    captions = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "description": ["A spiral galaxy.", "A merging pair."],
            "raw_response": ["A spiral galaxy.", "A merging pair."],
            "word_count": [3, 3],
        }
    )
    captions.to_json(base_output_root / "q_qwen_captions.jsonl", orient="records", lines=True)

    common_manifest = build_manifest(_manifest_source(), {}, seed=7)
    common_manifest = common_manifest[common_manifest["object_id"].isin(["a", "c"])]
    write_manifest(common_manifest, base_output_root / "common_manifest.parquet")

    with pytest.raises(ValueError, match="cache miss"):
        load_cached_common_set(base_output_root, source)


def _embedding_policy() -> NormalizationPolicy:
    return NormalizationPolicy(required=True, atol=1e-3)


def test_load_cached_text_embedding_caches_round_trips(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    base_output_root.mkdir(parents=True)
    policy = _embedding_policy()

    r_oai = ingest_released_embeddings(
        pd.DataFrame(
            {
                "object_id": ["a"],
                "summary": ["A visible spiral galaxy."],
                "summary_text_embedding": [[1.0, 0.0, 0.0]],
            }
        ),
        normalization_policy=policy,
    )
    write_embedding_cache(r_oai, base_output_root / "r_oai_embeddings.parquet", policy)

    r_qwen = ingest_released_embeddings(
        pd.DataFrame(
            {
                "object_id": ["a"],
                "summary": ["A visible spiral galaxy."],
                "summary_text_embedding": [[0.0, 1.0, 0.0]],
            }
        ),
        normalization_policy=policy,
    )
    write_embedding_cache(r_qwen, base_output_root / "r_qwen_embeddings.parquet", policy)

    spec = EmbeddingSpec(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="0" * 40,
        dimension=3,
        normalize=True,
        max_length=8192,
        query_instruction="Given an astronomical search query, retrieve matches.",
        batch_size=8,
    )
    q_qwen = embedding_frame(["a"], ["A qwen caption."], np.array([[0.0, 0.0, 1.0]]), "document", spec)
    write_embedding_cache(q_qwen, base_output_root / "q_qwen_embeddings.parquet", spec.normalization_policy)

    text_embedding_spec = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dimension": spec.dimension,
        "normalize": spec.normalize,
        "normalization_atol": 1e-3,
        "pooling": "last_token",
        "max_length": spec.max_length,
        "document_instruction": None,
        "query_instruction": spec.query_instruction,
        "batch_size": spec.batch_size,
    }

    loaded_r_oai, loaded_r_qwen, loaded_q_qwen, loaded_spec = load_cached_text_embedding_caches(
        base_output_root, text_embedding_spec
    )

    assert loaded_r_oai["object_id"].tolist() == ["a"]
    assert loaded_r_qwen["object_id"].tolist() == ["a"]
    assert loaded_q_qwen["object_id"].tolist() == ["a"]
    assert loaded_spec.query_instruction == spec.query_instruction


def test_load_cached_text_embedding_caches_missing_file_fails_loudly(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    base_output_root.mkdir(parents=True)
    text_embedding_spec = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "0" * 40,
        "dimension": 3,
        "normalize": True,
        "normalization_atol": 1e-3,
        "pooling": "last_token",
        "max_length": 8192,
        "document_instruction": None,
        "query_instruction": "Given an astronomical search query, retrieve matches.",
        "batch_size": 8,
    }

    with pytest.raises(FileNotFoundError, match="Cache-only mode requires an existing embedding cache"):
        load_cached_text_embedding_caches(base_output_root, text_embedding_spec)


def test_load_cached_text_embedding_caches_detects_fingerprint_mismatch(tmp_path) -> None:
    base_output_root = tmp_path / "phase3_10k_v1"
    base_output_root.mkdir(parents=True)
    policy = _embedding_policy()

    r_oai = ingest_released_embeddings(
        pd.DataFrame(
            {
                "object_id": ["a"],
                "summary": ["A visible spiral galaxy."],
                "summary_text_embedding": [[1.0, 0.0, 0.0]],
            }
        ),
        normalization_policy=policy,
    )
    path = base_output_root / "r_oai_embeddings.parquet"
    write_embedding_cache(r_oai, path, policy)
    tampered = pd.read_parquet(path)
    tampered.at[0, "text"] = "A tampered caption."
    tampered.to_parquet(path, index=False)

    text_embedding_spec = {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "0" * 40,
        "dimension": 3,
        "normalize": True,
        "normalization_atol": 1e-3,
        "pooling": "last_token",
        "max_length": 8192,
        "document_instruction": None,
        "query_instruction": "Given an astronomical search query, retrieve matches.",
        "batch_size": 8,
    }

    with pytest.raises(ValueError, match="cache miss"):
        load_cached_text_embedding_caches(base_output_root, text_embedding_spec)
