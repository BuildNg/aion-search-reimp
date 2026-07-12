from pathlib import Path

import pytest
import yaml

from aion_reimp.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_current_configs_validate() -> None:
    assert load_config(ROOT / "configs" / "phase0_reference.yaml")["kind"] == "phase0_reference"
    assert load_config(ROOT / "configs" / "phase1_open_text.yaml")["kind"] == "phase1_open_text"


def test_unknown_key_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase0_reference.yaml").read_text(encoding="utf-8"))
    data["run"]["mystery"] = 1
    with pytest.raises(ConfigError, match="Unknown run keys"):
        validate_config(data)


def test_document_instruction_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase1_open_text.yaml").read_text(encoding="utf-8"))
    data["text_embedding"]["document_instruction"] = "query-like instruction"
    with pytest.raises(ConfigError, match="document_instruction"):
        validate_config(data)


def test_pooling_must_be_last_token() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase1_open_text.yaml").read_text(encoding="utf-8"))
    data["text_embedding"]["pooling"] = "mean"
    with pytest.raises(ConfigError, match="last_token"):
        validate_config(data)


def test_phase_specific_top_level_key_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase1_open_text.yaml").read_text(encoding="utf-8"))
    data["reference_model"] = {}
    with pytest.raises(ConfigError, match="Unknown top-level keys"):
        validate_config(data)
