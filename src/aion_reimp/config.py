"""Strict configuration loading for current Phase 0/1 commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when a config is incomplete or contains unused keys."""


def _section(
    data: Mapping[str, Any],
    name: str,
    allowed: Iterable[str],
    required: Iterable[str],
) -> Dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    allowed_set = set(allowed)
    required_set = set(required)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(required_set - set(value))
    if unknown:
        raise ConfigError(f"Unknown {name} keys: {unknown}")
    if missing:
        raise ConfigError(f"Missing {name} keys: {missing}")
    return dict(value)


def _require_commit(value: Any, field: str) -> None:
    text = str(value)
    if len(text) != 40 or any(ch not in "0123456789abcdef" for ch in text.lower()):
        raise ConfigError(f"{field} must be a 40-character commit SHA, got {value!r}")


def validate_config(data: Mapping[str, Any]) -> Dict[str, Any]:
    common_top = {
        "schema_version",
        "kind",
        "run",
        "queries",
    }
    if data.get("schema_version") != 1:
        raise ConfigError("schema_version must equal 1")

    kind = data.get("kind")
    if kind not in {"phase0_reference", "phase1_open_text"}:
        raise ConfigError(f"Unsupported config kind: {kind!r}")
    kind_top = (
        {"reference_model", "reference_gate", "training_data", "benchmarks"}
        if kind == "phase0_reference"
        else {"captioning", "text_embedding", "released_text", "caption_audit"}
    )
    allowed_top = common_top | kind_top
    unknown = sorted(set(data) - allowed_top)
    missing_top = sorted(allowed_top - set(data))
    if unknown:
        raise ConfigError(f"Unknown top-level keys for {kind}: {unknown}")
    if missing_top:
        raise ConfigError(f"Missing top-level keys for {kind}: {missing_top}")

    _section(data, "run", {"id", "output_root", "seed"}, {"id", "output_root", "seed"})
    queries = _section(
        data,
        "queries",
        {"file", "openai_model", "openai_cache", "qwen_instruction"},
        {"file", "openai_model", "openai_cache", "qwen_instruction"},
    )
    if queries["openai_model"] != "text-embedding-3-large":
        raise ConfigError("R-OAI query model must be text-embedding-3-large")

    if kind == "phase0_reference":
        reference = _section(
            data,
            "reference_model",
            {"repo_id", "revision", "config_file", "weights_file", "orig_repo"},
            {"repo_id", "revision", "config_file", "weights_file", "orig_repo"},
        )
        _require_commit(reference["revision"], "reference_model.revision")
        gate = _section(
            data,
            "reference_gate",
            {"metric", "published_rounded_targets", "lens_top10_positives"},
            {"metric", "published_rounded_targets", "lens_top10_positives"},
        )
        if gate["metric"] != "ndcg@10":
            raise ConfigError("reference_gate.metric must be ndcg@10")
        targets = gate["published_rounded_targets"]
        if not isinstance(targets, dict) or set(targets) != {"spiral", "merger", "lens"}:
            raise ConfigError("reference_gate must define spiral, merger, and lens targets")
        if gate["lens_top10_positives"] != 2:
            raise ConfigError("reference_gate.lens_top10_positives must equal 2")
        training_data = _section(
            data,
            "training_data",
            {"repo_id", "revision", "split"},
            {"repo_id", "revision", "split"},
        )
        _require_commit(training_data["revision"], "training_data.revision")
        benchmarks = data.get("benchmarks")
        if not isinstance(benchmarks, list) or not benchmarks:
            raise ConfigError("benchmarks must be a non-empty list")
        for index, benchmark in enumerate(benchmarks):
            if not isinstance(benchmark, dict):
                raise ConfigError(f"benchmarks[{index}] must be a mapping")
            required = {"name", "repo_id", "revision"}
            unknown_benchmark = set(benchmark) - required
            missing_benchmark = required - set(benchmark)
            if unknown_benchmark or missing_benchmark:
                raise ConfigError(
                    f"benchmarks[{index}] unknown={sorted(unknown_benchmark)} "
                    f"missing={sorted(missing_benchmark)}"
                )
            _require_commit(benchmark["revision"], f"benchmarks[{index}].revision")
    else:
        captioning = _section(
            data,
            "captioning",
            {
                "model_id",
                "revision",
                "prompt_file",
                "dtype",
                "max_new_tokens",
                "do_sample",
            },
            {
                "model_id",
                "revision",
                "prompt_file",
                "dtype",
                "max_new_tokens",
                "do_sample",
            },
        )
        embedding = _section(
            data,
            "text_embedding",
            {
                "model_id",
                "revision",
                "dimension",
                "normalize",
                "pooling",
                "max_length",
                "document_instruction",
                "query_instruction",
            },
            {
                "model_id",
                "revision",
                "dimension",
                "normalize",
                "pooling",
                "max_length",
                "document_instruction",
                "query_instruction",
            },
        )
        released_text = _section(
            data,
            "released_text",
            {"repo_id", "revision", "split", "object_id_column", "text_column", "batch_size"},
            {"repo_id", "revision", "split", "object_id_column", "text_column", "batch_size"},
        )
        audit = _section(
            data,
            "caption_audit",
            {"repo_id", "revision", "split", "output_dir", "bootstrap_samples"},
            {"repo_id", "revision", "split", "output_dir", "bootstrap_samples"},
        )
        for section_name, section in (
            ("captioning", captioning),
            ("text_embedding", embedding),
            ("released_text", released_text),
            ("caption_audit", audit),
        ):
            _require_commit(section["revision"], f"{section_name}.revision")
        if embedding["document_instruction"] not in {None, ""}:
            raise ConfigError("text_embedding.document_instruction must be empty")
        if embedding["query_instruction"] != queries["qwen_instruction"]:
            raise ConfigError("Qwen query instruction must match queries.qwen_instruction")
        if embedding["dimension"] != 1024 or embedding["normalize"] is not True:
            raise ConfigError("Qwen embeddings must be normalized 1024-dimensional vectors")
        if embedding["pooling"] != "last_token":
            raise ConfigError("text_embedding.pooling must be last_token")
        if captioning["do_sample"] is not False:
            raise ConfigError("Phase 1 caption screen must use deterministic decoding")
        if not isinstance(released_text["batch_size"], int) or released_text["batch_size"] <= 0:
            raise ConfigError("released_text.batch_size must be a positive integer")

    return dict(data)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    return validate_config(raw)
