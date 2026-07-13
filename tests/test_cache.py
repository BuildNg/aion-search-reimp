import numpy as np
import pandas as pd
import pytest

from aion_reimp.cache import (
    NormalizationPolicy,
    cache_fingerprint,
    derive_fp32_normalized_cache,
    ingest_released_embeddings,
    sha256_text,
    validate_embedding_cache,
)


POLICY = NormalizationPolicy(required=True, atol=1e-3)


def test_released_embeddings_enter_common_document_schema() -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a"],
            "summary": ["A visible spiral galaxy."],
            "summary_text_embedding": [[1.0, 0.0, 0.0]],
        }
    )
    frame = ingest_released_embeddings(source, normalization_policy=POLICY)
    assert frame.loc[0, "role"] == "document"
    assert frame.loc[0, "instruction"] == ""
    assert frame.loc[0, "output_dim"] == 3


def test_released_openai_vector_with_small_norm_error_stays_verbatim() -> None:
    vector = [0.9996, 0.0, 0.0]
    source = pd.DataFrame(
        {
            "object_id": ["a"],
            "summary": ["A visible galaxy."],
            "summary_text_embedding": [vector],
        }
    )
    frame = ingest_released_embeddings(source, normalization_policy=POLICY)
    assert np.array_equal(
        np.asarray(frame.loc[0, "embedding"], dtype=np.float32),
        np.asarray(vector, dtype=np.float32),
    )
    assert bool(frame.loc[0, "normalized"]) is True


def test_document_with_query_instruction_fails() -> None:
    frame = pd.DataFrame(
        {
            "object_id": ["a"],
            "text": ["document"],
            "embedding": [[1.0, 0.0]],
            "role": ["document"],
            "model_id": ["model"],
            "revision": ["0" * 40],
            "instruction": ["retrieve something"],
            "instruction_hash": [sha256_text("retrieve something")],
            "source_checksum": [sha256_text("document")],
            "output_dim": [2],
            "normalized": [True],
        }
    )
    with pytest.raises(ValueError, match="query instruction"):
        validate_embedding_cache(frame, POLICY)


def test_validator_remeasures_norm_instead_of_trusting_flag() -> None:
    frame = pd.DataFrame(
        {
            "object_id": ["a"],
            "text": ["document"],
            "embedding": [[2.0, 0.0]],
            "role": ["document"],
            "model_id": ["model"],
            "revision": ["0" * 40],
            "instruction": [""],
            "instruction_hash": [sha256_text("")],
            "source_checksum": [sha256_text("document")],
            "output_dim": [2],
            "normalized": [True],
        }
    )
    with pytest.raises(ValueError, match="contradicts measured norm"):
        validate_embedding_cache(frame, POLICY)


def test_fingerprint_changes_when_vector_payload_changes() -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a"],
            "summary": ["A visible spiral galaxy."],
            "summary_text_embedding": [[1.0, 0.0, 0.0]],
        }
    )
    first = ingest_released_embeddings(source, normalization_policy=POLICY)
    second = first.copy()
    second.at[0, "embedding"] = [0.0, 1.0, 0.0]
    assert cache_fingerprint(first) != cache_fingerprint(second)


def test_derived_normalized_cache_is_new_and_records_lineage(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a"],
            "text": ["document"],
            "embedding": [[0.999, 0.001]],
            "role": ["document"],
            "model_id": ["model"],
            "revision": ["0" * 40],
            "instruction": [""],
            "instruction_hash": [sha256_text("")],
            "source_checksum": [sha256_text("document")],
            "output_dim": [2],
            "normalized": [True],
        }
    )
    source_path = tmp_path / "source.parquet"
    output_path = tmp_path / "derived" / "source.parquet"
    source.to_parquet(source_path, index=False)
    source_before = source_path.read_bytes()

    fingerprint = derive_fp32_normalized_cache(source_path, output_path, POLICY)

    assert source_path.read_bytes() == source_before
    derived = pd.read_parquet(output_path)
    validate_embedding_cache(derived, POLICY)
    assert fingerprint != cache_fingerprint(source)
    meta = pd.read_json(output_path.with_suffix(".parquet.meta.json"), typ="series")
    assert meta["source_fingerprint"] == cache_fingerprint(source)
    assert meta["transform"] == "fp32_l2_renormalize_v1"
