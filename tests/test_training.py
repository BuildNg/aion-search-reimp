import numpy as np
import pandas as pd
import pytest

from aion_reimp.training import assemble_condition_rows


def test_condition_join_is_one_to_one_and_manifest_complete() -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "split": ["train", "validation"],
            "image_embedding": [np.ones(3).tolist(), np.zeros(3).tolist()],
        }
    )
    text = pd.DataFrame(
        {
            "object_id": ["b", "a", "extra"],
            "embedding": [np.ones(2).tolist(), np.zeros(2).tolist(), np.ones(2).tolist()],
        }
    )
    joined = assemble_condition_rows(source, text)
    assert joined["object_id"].tolist() == ["a", "b"]


def test_condition_join_rejects_missing_text() -> None:
    source = pd.DataFrame(
        {
            "object_id": ["a", "b"],
            "split": ["train", "validation"],
            "image_embedding": [np.ones(3).tolist(), np.zeros(3).tolist()],
        }
    )
    text = pd.DataFrame({"object_id": ["a"], "embedding": [np.ones(2).tolist()]})
    with pytest.raises(ValueError, match="missing manifest objects"):
        assemble_condition_rows(source, text)
