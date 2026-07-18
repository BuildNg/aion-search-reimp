from pathlib import Path

import pytest
import yaml

from spec_probes.config import SpecProbesConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "phase6_probes.yaml"


def _load_raw():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_phase6_probes_config_validates() -> None:
    config = load_config(CONFIG_PATH)
    assert config["kind"] == "phase6_probes"
    assert len(config["encoders"]) == 3


def test_unknown_top_level_key_is_rejected() -> None:
    data = _load_raw()
    data["mystery"] = 1
    with pytest.raises(SpecProbesConfigError, match="Unknown top-level keys"):
        validate_config(data)


def test_unknown_source_data_key_is_rejected() -> None:
    data = _load_raw()
    data["source_data"]["mystery"] = 1
    with pytest.raises(SpecProbesConfigError, match="Unknown source_data keys"):
        validate_config(data)


def test_zwarn_filter_must_equal_zero() -> None:
    data = _load_raw()
    data["labels"]["zwarn_filter_value"] = 1
    with pytest.raises(SpecProbesConfigError, match="zwarn_filter_value must equal 0"):
        validate_config(data)


def test_streaming_must_be_true() -> None:
    data = _load_raw()
    data["source_data"]["streaming"] = False
    with pytest.raises(SpecProbesConfigError, match="streaming must be true"):
        validate_config(data)


def test_run_device_must_be_cuda_or_cpu() -> None:
    data = _load_raw()
    data["run"]["device"] = "tpu"
    with pytest.raises(SpecProbesConfigError, match="device must be"):
        validate_config(data)


def test_run_preflight_report_must_be_non_empty() -> None:
    data = _load_raw()
    data["run"]["preflight_report"] = ""
    with pytest.raises(SpecProbesConfigError, match="preflight_report"):
        validate_config(data)


def test_spectral_class_enabled_true_is_rejected() -> None:
    """SPECTYPE requires a separate DESI VAC crossmatch deliverable that
    does not exist yet -- the config must only accept options the code
    executes, so `enabled: true` is rejected rather than silently accepted."""
    data = _load_raw()
    data["labels"]["spectral_class"]["enabled"] = True
    with pytest.raises(SpecProbesConfigError, match="spectral_class.enabled must be false"):
        validate_config(data)


def test_spectral_class_classes_must_match_exactly() -> None:
    data = _load_raw()
    data["labels"]["spectral_class"]["classes"] = ["GALAXY", "QSO"]
    with pytest.raises(SpecProbesConfigError, match="spectral_class.classes must contain exactly"):
        validate_config(data)


def test_split_seeds_must_be_non_empty() -> None:
    data = _load_raw()
    data["split"]["seeds"] = []
    with pytest.raises(SpecProbesConfigError, match="split.seeds must be a non-empty list"):
        validate_config(data)


def test_split_seeds_must_not_contain_duplicates() -> None:
    data = _load_raw()
    data["split"]["seeds"] = [1, 1]
    with pytest.raises(SpecProbesConfigError, match="must not contain duplicates"):
        validate_config(data)


def test_encoders_must_be_exactly_three() -> None:
    data = _load_raw()
    data["encoders"] = [data["encoders"][0], data["encoders"][1].copy()]
    with pytest.raises(SpecProbesConfigError, match="exactly three encoder specs"):
        validate_config(data)


def test_duplicate_encoder_kind_is_rejected() -> None:
    data = _load_raw()
    pca = data["encoders"][2]
    data["encoders"] = [data["encoders"][0], data["encoders"][0].copy(), pca]
    with pytest.raises(SpecProbesConfigError, match="Duplicate encoder kind"):
        validate_config(data)


def test_encoder_revision_must_be_a_commit_sha() -> None:
    data = _load_raw()
    data["encoders"][0]["revision"] = "main"
    with pytest.raises(SpecProbesConfigError, match="commit SHA"):
        validate_config(data)


def test_pca_baseline_requires_its_own_fields() -> None:
    data = _load_raw()
    del data["encoders"][2]["resample_n_pixels"]
    with pytest.raises(SpecProbesConfigError, match="pca_baseline\\) has the wrong fields"):
        validate_config(data)


def test_neural_encoder_requires_batch_size_and_dtype() -> None:
    data = _load_raw()
    del data["encoders"][0]["batch_size"]
    with pytest.raises(SpecProbesConfigError, match="wrong fields"):
        validate_config(data)


def test_neural_encoder_batch_size_must_be_positive() -> None:
    data = _load_raw()
    data["encoders"][0]["batch_size"] = 0
    with pytest.raises(SpecProbesConfigError, match="batch_size must be a positive integer"):
        validate_config(data)


def test_neural_encoder_dtype_must_be_known() -> None:
    data = _load_raw()
    data["encoders"][0]["dtype"] = "int8"
    with pytest.raises(SpecProbesConfigError, match="dtype must be one of"):
        validate_config(data)


def test_neural_encoder_half_precision_is_rejected_until_implemented() -> None:
    data = _load_raw()
    data["encoders"][0]["dtype"] = "float16"
    with pytest.raises(SpecProbesConfigError, match="dtype must be one of"):
        validate_config(data)


def test_neural_encoder_package_must_match_loading_route() -> None:
    data = _load_raw()
    data["encoders"][0]["package"] = "huggingface_hub"
    with pytest.raises(SpecProbesConfigError, match="package must be 'aion'"):
        validate_config(data)


def test_aion_encoder_requires_num_encoder_tokens() -> None:
    data = _load_raw()
    del data["encoders"][0]["num_encoder_tokens"]
    with pytest.raises(SpecProbesConfigError, match="wrong fields"):
        validate_config(data)


def test_specformer_encoder_rejects_num_encoder_tokens() -> None:
    data = _load_raw()
    data["encoders"][1]["num_encoder_tokens"] = 600
    with pytest.raises(SpecProbesConfigError, match="wrong fields"):
        validate_config(data)


def test_knn_metric_must_be_cosine() -> None:
    data = _load_raw()
    data["probes"]["knn"]["metric"] = "euclidean"
    with pytest.raises(SpecProbesConfigError, match="must be 'cosine'"):
        validate_config(data)


def test_knn_k_must_be_a_positive_integer() -> None:
    data = _load_raw()
    data["probes"]["knn"]["k"] = 0
    with pytest.raises(SpecProbesConfigError, match="k must be a positive integer"):
        validate_config(data)


def test_probes_scaling_must_be_standardize() -> None:
    data = _load_raw()
    data["probes"]["scaling"] = "minmax"
    with pytest.raises(SpecProbesConfigError, match="scaling must be 'standardize'"):
        validate_config(data)


def test_probes_cv_folds_must_be_at_least_two() -> None:
    data = _load_raw()
    data["probes"]["cv_folds"] = 1
    with pytest.raises(SpecProbesConfigError, match="cv_folds must be an integer >= 2"):
        validate_config(data)


def test_ridge_alpha_grid_must_be_non_empty() -> None:
    data = _load_raw()
    data["probes"]["linear"]["ridge_alpha_grid"] = []
    with pytest.raises(SpecProbesConfigError, match="ridge_alpha_grid must be a non-empty list"):
        validate_config(data)


def test_ridge_alpha_grid_rejects_duplicates() -> None:
    data = _load_raw()
    data["probes"]["linear"]["ridge_alpha_grid"] = [1.0, 1.0]
    with pytest.raises(SpecProbesConfigError, match="must not contain duplicates"):
        validate_config(data)


def test_logistic_c_grid_rejects_non_positive_values() -> None:
    data = _load_raw()
    data["probes"]["linear"]["logistic_c_grid"] = [1.0, -1.0]
    with pytest.raises(SpecProbesConfigError, match="logistic_c_grid must contain only positive numbers"):
        validate_config(data)


def test_train_ratio_must_be_between_zero_and_one() -> None:
    data = _load_raw()
    data["split"]["train_ratio"] = 1.5
    with pytest.raises(SpecProbesConfigError, match="train_ratio must be between zero and one"):
        validate_config(data)
