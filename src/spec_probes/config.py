"""Strict configuration loading for the Phase 6 frozen spectrum-encoder probes.

A separate schema from aion_reimp.config: Phase 6 probes are a bounded,
independent package (architecture.md decision 12) and must not extend the
aion_reimp config's ``kind`` dispatch. Unknown keys fail, matching the
strictness convention used across the rest of this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class SpecProbesConfigError(ValueError):
    """Raised when a Phase 6 probe config is incomplete or contains unused keys."""


ENCODER_KINDS = {"aion_spectrumcodec", "astroclip_specformer", "pca_baseline"}
SPECTYPE_CLASSES = ("GALAXY", "QSO", "STAR")
# The adapters currently run the vendored/upstream models in float32. Do
# not advertise half-precision until model casting/autocast is implemented
# and exercised against both real checkpoints.
DTYPES = {"float32"}


def _section(
    data: Mapping[str, Any],
    name: str,
    allowed: Iterable[str],
    required: Iterable[str],
) -> Dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise SpecProbesConfigError(f"{name} must be a mapping")
    allowed_set = set(allowed)
    required_set = set(required)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise SpecProbesConfigError(f"Unknown {name} keys: {unknown}")
    if missing:
        raise SpecProbesConfigError(f"Missing {name} keys: {missing}")
    return dict(value)


def _require_commit(value: Any, field: str) -> None:
    text = str(value)
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise SpecProbesConfigError(f"{field} must be a 40-character commit SHA, got {value!r}")


def _validate_run(data: Mapping[str, Any]) -> Dict[str, Any]:
    run = _section(
        data,
        "run",
        {"id", "output_root", "seed", "device", "preflight_report"},
        {"id", "output_root", "seed", "device", "preflight_report"},
    )
    if run["device"] not in {"cuda", "cpu"}:
        raise SpecProbesConfigError("run.device must be 'cuda' or 'cpu'")
    if not str(run["preflight_report"]).strip():
        raise SpecProbesConfigError("run.preflight_report must be a non-empty path")
    return run


def _validate_source_data(data: Mapping[str, Any]) -> Dict[str, Any]:
    source = _section(
        data,
        "source_data",
        {
            "repo_id",
            "revision",
            "split",
            "streaming",
            "sample_size",
            "object_id_column",
            "spectrum_column",
            "flux_field",
            "wave_field",
            "ivar_field",
            "mask_field",
            "redshift_column",
            "zwarn_column",
        },
        {
            "repo_id",
            "revision",
            "split",
            "streaming",
            "sample_size",
            "object_id_column",
            "spectrum_column",
            "flux_field",
            "wave_field",
            "ivar_field",
            "mask_field",
            "redshift_column",
            "zwarn_column",
        },
    )
    _require_commit(source["revision"], "source_data.revision")
    if source["streaming"] is not True:
        raise SpecProbesConfigError(
            "source_data.streaming must be true; the probe sample is streamed, never bulk-downloaded"
        )
    if (
        not isinstance(source["sample_size"], int)
        or isinstance(source["sample_size"], bool)
        or source["sample_size"] <= 0
    ):
        raise SpecProbesConfigError("source_data.sample_size must be a positive integer")
    return source


def _validate_labels(data: Mapping[str, Any]) -> Dict[str, Any]:
    labels = _section(data, "labels", {"zwarn_filter_value", "spectral_class"}, {"zwarn_filter_value", "spectral_class"})
    if labels["zwarn_filter_value"] != 0:
        raise SpecProbesConfigError("labels.zwarn_filter_value must equal 0 (the ZWARN == 0 filter)")
    spectral_class = labels["spectral_class"]
    if not isinstance(spectral_class, dict) or set(spectral_class) != {"enabled", "classes"}:
        raise SpecProbesConfigError("labels.spectral_class has the wrong fields")
    if not isinstance(spectral_class["enabled"], bool):
        raise SpecProbesConfigError("labels.spectral_class.enabled must be a boolean")
    classes = spectral_class["classes"]
    if not isinstance(classes, list) or set(classes) != set(SPECTYPE_CLASSES):
        raise SpecProbesConfigError(f"labels.spectral_class.classes must contain exactly {sorted(SPECTYPE_CLASSES)}")
    if spectral_class["enabled"]:
        # No streamed MMU DESI column and no join implementation exist yet
        # (verified 2026-07-18 against the dataset schema and the MMU
        # builder script -- see spectra_data.py); SPECTYPE requires a
        # separate DESI VAC crossmatch deliverable. Reject rather than
        # silently accept an option the code cannot execute.
        raise SpecProbesConfigError(
            "labels.spectral_class.enabled must be false: SPECTYPE is not present in the streamed "
            "MultimodalUniverse/desi records and no crossmatch join is implemented in this package "
            "(a DESI VAC crossmatch, keyed by TARGETID == this dataset's object_id, is a separate, "
            "not-yet-scoped deliverable). Only the redshift probes run."
        )
    return labels


def _validate_encoders(data: Mapping[str, Any]) -> None:
    encoders = data.get("encoders")
    if not isinstance(encoders, list) or len(encoders) != 3:
        raise SpecProbesConfigError("encoders must be a list of exactly three encoder specs")
    seen_kinds = set()
    seen_names = set()
    for index, encoder in enumerate(encoders):
        if not isinstance(encoder, dict):
            raise SpecProbesConfigError(f"encoders[{index}] must be a mapping")
        kind = encoder.get("kind")
        if kind not in ENCODER_KINDS:
            raise SpecProbesConfigError(f"encoders[{index}].kind must be one of {sorted(ENCODER_KINDS)}")
        if kind in seen_kinds:
            raise SpecProbesConfigError(f"Duplicate encoder kind: {kind}")
        seen_kinds.add(kind)
        name = encoder.get("name")
        if not name or name in seen_names:
            raise SpecProbesConfigError(f"encoders[{index}].name must be unique and non-empty")
        seen_names.add(name)
        if kind == "pca_baseline":
            required = {"name", "kind", "n_components", "continuum_normalize", "resample_n_pixels"}
            if set(encoder) != required:
                raise SpecProbesConfigError(f"encoders[{index}] (pca_baseline) has the wrong fields")
            if not isinstance(encoder["n_components"], int) or encoder["n_components"] <= 0:
                raise SpecProbesConfigError("pca_baseline.n_components must be a positive integer")
            if encoder["continuum_normalize"] is not True:
                raise SpecProbesConfigError("pca_baseline.continuum_normalize must be true")
            if not isinstance(encoder["resample_n_pixels"], int) or encoder["resample_n_pixels"] <= 0:
                raise SpecProbesConfigError("pca_baseline.resample_n_pixels must be a positive integer")
            continue

        base_required = {"name", "kind", "package", "repo_id", "revision", "output_dim", "pooling", "batch_size", "dtype"}
        required = base_required | ({"num_encoder_tokens"} if kind == "aion_spectrumcodec" else set())
        if set(encoder) != required:
            raise SpecProbesConfigError(f"encoders[{index}] ({kind}) has the wrong fields")
        _require_commit(encoder["revision"], f"encoders[{index}].revision")
        if not isinstance(encoder["output_dim"], int) or encoder["output_dim"] <= 0:
            raise SpecProbesConfigError(f"encoders[{index}].output_dim must be a positive integer")
        if encoder["pooling"] != "mean":
            raise SpecProbesConfigError(f"encoders[{index}].pooling must be 'mean'")
        expected_package = "aion" if kind == "aion_spectrumcodec" else "huggingface_hub"
        if encoder["package"] != expected_package:
            raise SpecProbesConfigError(
                f"encoders[{index}].package must be {expected_package!r} for kind {kind!r}"
            )
        if not isinstance(encoder["batch_size"], int) or encoder["batch_size"] <= 0:
            raise SpecProbesConfigError(f"encoders[{index}].batch_size must be a positive integer")
        if encoder["dtype"] not in DTYPES:
            raise SpecProbesConfigError(f"encoders[{index}].dtype must be one of {sorted(DTYPES)}")
        if kind == "aion_spectrumcodec":
            tokens = encoder["num_encoder_tokens"]
            if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
                raise SpecProbesConfigError(f"encoders[{index}].num_encoder_tokens must be a positive integer")
    if seen_kinds != ENCODER_KINDS:
        raise SpecProbesConfigError(f"encoders must cover exactly {sorted(ENCODER_KINDS)}")


def _validate_probes(data: Mapping[str, Any]) -> None:
    probes = _section(data, "probes", {"scaling", "cv_folds", "linear", "knn"}, {"scaling", "cv_folds", "linear", "knn"})
    if probes["scaling"] != "standardize":
        raise SpecProbesConfigError("probes.scaling must be 'standardize'")
    if not isinstance(probes["cv_folds"], int) or probes["cv_folds"] < 2:
        raise SpecProbesConfigError("probes.cv_folds must be an integer >= 2")

    linear = probes["linear"]
    if not isinstance(linear, dict) or set(linear) != {"ridge_alpha_grid", "logistic_c_grid", "logistic_max_iter"}:
        raise SpecProbesConfigError("probes.linear has the wrong fields")
    for grid_name in ("ridge_alpha_grid", "logistic_c_grid"):
        grid = linear[grid_name]
        if not isinstance(grid, list) or not grid:
            raise SpecProbesConfigError(f"probes.linear.{grid_name} must be a non-empty list")
        if any((not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0) for value in grid):
            raise SpecProbesConfigError(f"probes.linear.{grid_name} must contain only positive numbers")
        if len(set(grid)) != len(grid):
            raise SpecProbesConfigError(f"probes.linear.{grid_name} must not contain duplicates")
    if not isinstance(linear["logistic_max_iter"], int) or linear["logistic_max_iter"] <= 0:
        raise SpecProbesConfigError("probes.linear.logistic_max_iter must be a positive integer")

    knn = probes["knn"]
    if not isinstance(knn, dict) or set(knn) != {"k", "metric"}:
        raise SpecProbesConfigError("probes.knn has the wrong fields")
    if not isinstance(knn["k"], int) or knn["k"] <= 0:
        raise SpecProbesConfigError("probes.knn.k must be a positive integer")
    if knn["metric"] != "cosine":
        raise SpecProbesConfigError("probes.knn.metric must be 'cosine'")


def _validate_metrics(data: Mapping[str, Any]) -> None:
    metrics = _section(
        data, "metrics", {"catastrophic_outlier_threshold"}, {"catastrophic_outlier_threshold"}
    )
    if float(metrics["catastrophic_outlier_threshold"]) <= 0:
        raise SpecProbesConfigError("metrics.catastrophic_outlier_threshold must be positive")


def validate_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    top_allowed = {
        "schema_version",
        "kind",
        "run",
        "source_data",
        "labels",
        "split",
        "encoders",
        "probes",
        "metrics",
    }
    if data.get("schema_version") != 1:
        raise SpecProbesConfigError("schema_version must equal 1")
    if data.get("kind") != "phase6_probes":
        raise SpecProbesConfigError(f"Unsupported config kind: {data.get('kind')!r}")
    unknown_top = sorted(set(data) - top_allowed)
    missing_top = sorted(top_allowed - set(data))
    if unknown_top:
        raise SpecProbesConfigError(f"Unknown top-level keys: {unknown_top}")
    if missing_top:
        raise SpecProbesConfigError(f"Missing top-level keys: {missing_top}")

    _validate_run(data)
    _validate_source_data(data)
    _validate_labels(data)

    split = _section(data, "split", {"train_ratio", "seeds"}, {"train_ratio", "seeds"})
    if not 0.0 < float(split["train_ratio"]) < 1.0:
        raise SpecProbesConfigError("split.train_ratio must be between zero and one")
    seeds = split["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise SpecProbesConfigError("split.seeds must be a non-empty list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
        raise SpecProbesConfigError("split.seeds must contain only integers")
    if len(set(seeds)) != len(seeds):
        raise SpecProbesConfigError("split.seeds must not contain duplicates")

    _validate_encoders(data)
    _validate_probes(data)
    _validate_metrics(data)

    return dict(data)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise SpecProbesConfigError("Config root must be a mapping")
    return validate_config(raw)
