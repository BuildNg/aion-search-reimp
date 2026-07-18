from pathlib import Path

import pytest
import yaml

from aion_reimp.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_current_configs_validate() -> None:
    assert load_config(ROOT / "configs" / "phase0_reference.yaml")["kind"] == "phase0_reference"
    assert load_config(ROOT / "configs" / "phase1.yaml")["kind"] == "phase1"
    assert load_config(ROOT / "configs" / "phase2_smoke.yaml")["kind"] == "phase2_smoke"
    assert load_config(ROOT / "configs" / "phase3_10k.yaml")["kind"] == "phase3_10k"
    assert (
        load_config(ROOT / "configs" / "phase3_10k_seedext.yaml")["kind"]
        == "phase3_10k_seedext"
    )


def test_seedext_run_id_and_seeds_differ_from_base_pilot() -> None:
    base = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    seedext = load_config(ROOT / "configs" / "phase3_10k_seedext.yaml")
    assert seedext["run"]["id"] != base["run"]["id"]
    assert seedext["run"]["seed"] == base["run"]["seed"]
    assert set(seedext["seeds"]).isdisjoint(set(base["seeds"]))
    assert seedext["source_run"]["run_id"] == base["run"]["id"]
    assert set(seedext["source_run"]["reused_seeds"]) == set(base["seeds"])


def test_seedext_requires_only_one_seed() -> None:
    data = yaml.safe_load(
        (ROOT / "configs" / "phase3_10k_seedext.yaml").read_text(encoding="utf-8")
    )
    data["seeds"] = [99]
    assert validate_config(data)["seeds"] == [99]


def test_seedext_rejects_seeds_that_overlap_the_reused_run() -> None:
    data = yaml.safe_load(
        (ROOT / "configs" / "phase3_10k_seedext.yaml").read_text(encoding="utf-8")
    )
    data["seeds"] = [45, 13]
    with pytest.raises(ConfigError, match="must be disjoint from source_run.reused_seeds"):
        validate_config(data)


def test_seedext_requires_source_run_section() -> None:
    data = yaml.safe_load(
        (ROOT / "configs" / "phase3_10k_seedext.yaml").read_text(encoding="utf-8")
    )
    del data["source_run"]
    with pytest.raises(ConfigError, match="Missing top-level keys"):
        validate_config(data)


def test_seedext_rejects_empty_seed_list() -> None:
    data = yaml.safe_load(
        (ROOT / "configs" / "phase3_10k_seedext.yaml").read_text(encoding="utf-8")
    )
    data["seeds"] = []
    with pytest.raises(ConfigError, match="at least one seed"):
        validate_config(data)


def test_phase3_requires_ten_thousand_sample_size() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    data["source_data"]["sample_size"] = 1000
    with pytest.raises(ConfigError, match="sample_size must equal 10000"):
        validate_config(data)


def test_phase3_requires_at_least_three_distinct_seeds() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    data["seeds"] = [13, 13]
    with pytest.raises(ConfigError, match="at least three seeds"):
        validate_config(data)


def test_phase3_rejects_duplicate_seeds() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    data["seeds"] = [13, 21, 13]
    with pytest.raises(ConfigError, match="must not contain duplicates"):
        validate_config(data)


def test_phase3_requires_gz_decals_and_lens_benchmarks() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    data["benchmarks"] = [data["benchmarks"][0]]
    with pytest.raises(ConfigError, match="must name exactly gz_decals and lens"):
        validate_config(data)


def test_gpt_reference_rejects_unpinned_alias() -> None:
    data = yaml.safe_load(
        (ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8")
    )
    data["captioners"]["gpt"]["model_id"] = "openai/gpt-4.1-mini"
    with pytest.raises(ConfigError, match="must pin"):
        validate_config(data)


def test_phase1_rejects_a_caption_enforcement_policy() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    data["caption_policy"] = {"max_words": 300}
    with pytest.raises(ConfigError, match="Unknown top-level keys"):
        validate_config(data)


def test_unknown_key_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase0_reference.yaml").read_text(encoding="utf-8"))
    data["run"]["mystery"] = 1
    with pytest.raises(ConfigError, match="Unknown run keys"):
        validate_config(data)


def test_document_instruction_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase2_smoke.yaml").read_text(encoding="utf-8"))
    data["text_embedding"]["document_instruction"] = "query-like instruction"
    with pytest.raises(ConfigError, match="document_instruction"):
        validate_config(data)


def test_pooling_must_be_last_token() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase2_smoke.yaml").read_text(encoding="utf-8"))
    data["text_embedding"]["pooling"] = "mean"
    with pytest.raises(ConfigError, match="last_token"):
        validate_config(data)


def test_text_embedding_batch_size_must_be_positive_int() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase3_10k.yaml").read_text(encoding="utf-8"))
    data["text_embedding"]["batch_size"] = 0
    with pytest.raises(ConfigError, match="batch_size must be positive"):
        validate_config(data)

    data["text_embedding"]["batch_size"] = "64"
    with pytest.raises(ConfigError, match="batch_size must be an integer"):
        validate_config(data)


def test_phase_specific_top_level_key_is_rejected() -> None:
    data = yaml.safe_load((ROOT / "configs" / "phase1.yaml").read_text(encoding="utf-8"))
    data["reference_model"] = {}
    with pytest.raises(ConfigError, match="Unknown top-level keys"):
        validate_config(data)
