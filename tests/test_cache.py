import numpy as np
import pandas as pd
import pytest

from aion_reimp.cache import ingest_released_embeddings, sha256_text, validate_embedding_cache


def test_released_embeddings_enter_common_document_schema() -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a"],
            "summary": ["A visible spiral galaxy."],
            "summary_text_embedding": [[1.0, 0.0, 0.0]],
        }
    )
    frame = ingest_released_embeddings(source)
    assert frame.loc[0, "role"] == "document"
    assert frame.loc[0, "instruction"] == ""
    assert frame.loc[0, "output_dim"] == 3


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
        validate_embedding_cache(frame)
