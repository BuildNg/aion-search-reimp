from pathlib import Path

import pytest

from aion_reimp.config import load_config
from aion_reimp.reference import (
    assert_reference_equivalence,
    load_author_reference,
    load_reimplemented_reference,
    resolve_released_files,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.cluster
def test_released_checkpoint_matches_reimplementation() -> None:
    config = load_config(ROOT / "configs" / "phase0_reference.yaml")
    spec = config["reference_model"]
    config_path, weights_path = resolve_released_files(
        spec["repo_id"], spec["revision"], spec["config_file"], spec["weights_file"]
    )
    reimplemented = load_reimplemented_reference(config_path, weights_path)
    author = load_author_reference(ROOT / spec["orig_repo"], config_path, weights_path)
    maxima = assert_reference_equivalence(
        reimplemented,
        author,
        seed=config["run"]["seed"],
    )
    assert all(value <= 1e-6 for value in maxima.values())
