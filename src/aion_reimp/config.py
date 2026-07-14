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
    common_top = {"schema_version", "kind", "run"}
    if data.get("schema_version") != 1:
        raise ConfigError("schema_version must equal 1")

    kind = data.get("kind")
    if kind not in {
        "phase0_reference",
        "phase1",
        "phase2_smoke",
    }:
        raise ConfigError(f"Unsupported config kind: {kind!r}")
    if kind == "phase0_reference":
        kind_top = {"queries", "reference_model", "reference_gate", "training_data", "benchmarks"}
    elif kind == "phase1":
        kind_top = {
            "benchmark",
            "captioners",
            "extractor",
            "artifacts",
            "cost",
            "audit",
        }
    else:
        kind_top = {
            "queries",
            "prerequisites",
            "source_data",
            "exclusions",
            "captioning",
            "text_embedding",
            "caches",
            "model",
            "training",
            "conditions",
        }
    allowed_top = common_top | kind_top
    unknown = sorted(set(data) - allowed_top)
    missing_top = sorted(allowed_top - set(data))
    if unknown:
        raise ConfigError(f"Unknown top-level keys for {kind}: {unknown}")
    if missing_top:
        raise ConfigError(f"Missing top-level keys for {kind}: {missing_top}")

    _section(data, "run", {"id", "output_root", "seed"}, {"id", "output_root", "seed"})
    queries = None
    if kind in {"phase0_reference", "phase2_smoke"}:
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
    elif kind == "phase1":
        benchmark = _section(
            data,
            "benchmark",
            {"repo_id", "revision", "split", "input_dir"},
            {"repo_id", "revision", "split", "input_dir"},
        )
        _require_commit(benchmark["revision"], "benchmark.revision")
        captioners = data.get("captioners")
        if not isinstance(captioners, dict) or set(captioners) != {"qwen", "gpt"}:
            raise ConfigError("captioners must define exactly qwen and gpt")
        qwen = captioners["qwen"]
        qwen_keys = {
            "model_id",
            "revision",
            "prompt_file",
            "dtype",
            "max_new_tokens",
            "do_sample",
        }
        if not isinstance(qwen, dict) or set(qwen) != qwen_keys:
            raise ConfigError("captioners.qwen has the wrong fields")
        _require_commit(qwen["revision"], "captioners.qwen.revision")
        if qwen["do_sample"] is not False:
            raise ConfigError("Phase 1 Qwen captioning must use deterministic decoding")
        gpt = captioners["gpt"]
        gpt_keys = {
            "model_id",
            "provider",
            "base_url",
            "api_key_env",
            "prompt_file",
            "temperature",
            "max_output_tokens",
            "image_detail",
        }
        if not isinstance(gpt, dict) or set(gpt) != gpt_keys:
            raise ConfigError("captioners.gpt has the wrong fields")
        if gpt["model_id"] != "openai/gpt-4.1-mini-2025-04-14":
            raise ConfigError("GPT reference must pin openai/gpt-4.1-mini-2025-04-14")
        if gpt["provider"] != "OpenAI" or float(gpt["temperature"]) != 0.0:
            raise ConfigError("GPT reference must pin OpenAI and temperature zero")
        if gpt["image_detail"] != "low":
            raise ConfigError("GPT reference image_detail must equal low")
        if Path(qwen["prompt_file"]) != Path(gpt["prompt_file"]):
            raise ConfigError("Qwen and GPT must use the same free-form caption prompt")

        extractor = _section(
            data,
            "extractor",
            {
                "model_id",
                "revision",
                "model_path",
                "prompt_file",
                "prompt_sha256",
                "structured_output_engine",
                "schema_variant",
                "dtype",
                "max_new_tokens",
                "enable_thinking",
                "calibration_file",
                "calibration_min_answer_accuracy",
                "max_error_rate",
            },
            {
                "model_id",
                "revision",
                "model_path",
                "prompt_file",
                "prompt_sha256",
                "structured_output_engine",
                "dtype",
                "max_new_tokens",
                "enable_thinking",
                "calibration_file",
                "calibration_min_answer_accuracy",
                "max_error_rate",
            },
        )
        if extractor.get("schema_variant", "flat") not in {"flat", "nested"}:
            raise ConfigError("extractor.schema_variant must be 'flat' or 'nested'")
        _require_commit(extractor["revision"], "extractor.revision")
        if extractor["model_id"] != "google/gemma-4-26B-A4B-it":
            raise ConfigError("Phase 1 extractor must pin google/gemma-4-26B-A4B-it")
        if extractor["enable_thinking"] is not False:
            raise ConfigError("Primary Gemma extractor must disable thinking")
        if extractor["structured_output_engine"] != "xgrammar":
            raise ConfigError("Phase 1 structured output engine must be xgrammar")
        prompt_hash = str(extractor["prompt_sha256"])
        if len(prompt_hash) != 64 or any(character not in "0123456789abcdef" for character in prompt_hash):
            raise ConfigError("extractor.prompt_sha256 must be a lowercase SHA-256 digest")
        if not 0.0 < float(extractor["calibration_min_answer_accuracy"]) <= 1.0:
            raise ConfigError(
                "extractor.calibration_min_answer_accuracy must be in (0, 1]"
            )
        if not 0.0 <= float(extractor["max_error_rate"]) < 1.0:
            raise ConfigError("extractor.max_error_rate must be in [0, 1)")

        _section(
            data,
            "artifacts",
            {
                "image_preflight",
                "qwen_descriptions",
                "gpt_descriptions",
                "gpt_usage",
                "gpt_cost",
            },
            {
                "image_preflight",
                "qwen_descriptions",
                "gpt_descriptions",
                "gpt_usage",
                "gpt_cost",
            },
        )
        audit = _section(
            data,
            "audit",
            {"bootstrap_samples", "primary_metric", "secondary_metric"},
            {"bootstrap_samples", "primary_metric", "secondary_metric"},
        )
        if not isinstance(audit["bootstrap_samples"], int) or audit["bootstrap_samples"] <= 0:
            raise ConfigError("audit.bootstrap_samples must be positive")
        if audit["primary_metric"] != "released_decision_path_overlap":
            raise ConfigError("Phase 1 primary metric must reproduce released path overlap")
        if audit["secondary_metric"] != "per_question_accuracy":
            raise ConfigError("Phase 1 secondary metric must be per-question accuracy")
        cost = _section(
            data,
            "cost",
            {
                "input_usd_per_million",
                "output_usd_per_million",
                "hard_cap_usd",
                "reserve_per_request_usd",
            },
            {
                "input_usd_per_million",
                "output_usd_per_million",
                "hard_cap_usd",
                "reserve_per_request_usd",
            },
        )
        if float(cost["hard_cap_usd"]) != 0.1:
            raise ConfigError("GPT reference hard_cap_usd must equal 0.10")
        if float(cost["reserve_per_request_usd"]) <= 0:
            raise ConfigError("GPT reference reserve_per_request_usd must be positive")
    else:
        prerequisites = _section(
            data,
            "prerequisites",
            {
                "phase0_reference_gate",
                "phase1_qwen_caption_audit_rows",
                "phase1_gpt_caption_audit_rows",
                "bootstrap_samples",
            },
            {
                "phase0_reference_gate",
                "phase1_qwen_caption_audit_rows",
                "phase1_gpt_caption_audit_rows",
                "bootstrap_samples",
            },
        )
        if not isinstance(prerequisites["bootstrap_samples"], int) or prerequisites[
            "bootstrap_samples"
        ] <= 0:
            raise ConfigError("prerequisites.bootstrap_samples must be positive")

        source = _section(
            data,
            "source_data",
            {
                "repo_id",
                "revision",
                "split",
                "sample_size",
                "train_ratio",
                "object_id_column",
                "survey_column",
                "ra_column",
                "dec_column",
                "image_column",
                "image_embedding_column",
                "released_text_column",
                "released_embedding_column",
            },
            {
                "repo_id",
                "revision",
                "split",
                "sample_size",
                "train_ratio",
                "object_id_column",
                "survey_column",
                "ra_column",
                "dec_column",
                "image_column",
                "image_embedding_column",
                "released_text_column",
                "released_embedding_column",
            },
        )
        _require_commit(source["revision"], "source_data.revision")
        if source["sample_size"] != 1000:
            raise ConfigError("Phase 2 source_data.sample_size must equal 1000")
        if not 0.0 < float(source["train_ratio"]) < 1.0:
            raise ConfigError("source_data.train_ratio must be between zero and one")

        exclusions = _section(
            data,
            "exclusions",
            {"radius_arcsec", "caption_screen_labels", "benchmark_coordinates"},
            {"radius_arcsec", "caption_screen_labels", "benchmark_coordinates"},
        )
        if float(exclusions["radius_arcsec"]) <= 0:
            raise ConfigError("exclusions.radius_arcsec must be positive")
        if not isinstance(exclusions["benchmark_coordinates"], dict) or set(
            exclusions["benchmark_coordinates"]
        ) != {"gz_decals", "lens"}:
            raise ConfigError("exclusions must name gz_decals and lens coordinate artifacts")

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
                "max_error_rate",
            },
            {
                "model_id",
                "revision",
                "prompt_file",
                "dtype",
                "max_new_tokens",
                "do_sample",
                "max_error_rate",
            },
        )
        _require_commit(captioning["revision"], "captioning.revision")
        if captioning["do_sample"] is not False:
            raise ConfigError("Phase 2 captioning must use deterministic decoding")
        if not 0.0 <= float(captioning["max_error_rate"]) < 1.0:
            raise ConfigError("captioning.max_error_rate must be in [0, 1)")

        embedding = _section(
            data,
            "text_embedding",
            {
                "model_id",
                "revision",
                "dimension",
                "normalize",
                "normalization_atol",
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
                "normalization_atol",
                "pooling",
                "max_length",
                "document_instruction",
                "query_instruction",
            },
        )
        _require_commit(embedding["revision"], "text_embedding.revision")
        if embedding["document_instruction"] not in {None, ""}:
            raise ConfigError("text_embedding.document_instruction must be empty")
        if embedding["query_instruction"] != queries["qwen_instruction"]:
            raise ConfigError("Qwen query instruction must match queries.qwen_instruction")
        if embedding["dimension"] != 1024 or embedding["normalize"] is not True:
            raise ConfigError("Qwen embeddings must be normalized 1024-dimensional vectors")
        if embedding["pooling"] != "last_token":
            raise ConfigError("text_embedding.pooling must be last_token")
        if float(embedding["normalization_atol"]) != 1e-3:
            raise ConfigError("text_embedding.normalization_atol must equal 0.001")

        _section(
            data,
            "caches",
            {"phase1_normalized_dir", "released_summary_qwen"},
            {"phase1_normalized_dir", "released_summary_qwen"},
        )
        model = _section(
            data,
            "model",
            {
                "image_input_dim",
                "embedding_dim",
                "image_hidden_dim",
                "text_hidden_dim",
                "dropout",
                "use_mean_embeddings",
                "temperature_parameterization",
                "temperature_initial_scale",
                "temperature_max_scale",
            },
            {
                "image_input_dim",
                "embedding_dim",
                "image_hidden_dim",
                "text_hidden_dim",
                "dropout",
                "use_mean_embeddings",
                "temperature_parameterization",
                "temperature_initial_scale",
                "temperature_max_scale",
            },
        )
        if model["temperature_parameterization"] != "log":
            raise ConfigError("model.temperature_parameterization must be log")
        if abs(float(model["temperature_initial_scale"]) - 1.0 / 0.07) > 1e-12:
            raise ConfigError("model.temperature_initial_scale must equal 1/0.07")
        if float(model["temperature_max_scale"]) != 100.0:
            raise ConfigError("model.temperature_max_scale must equal 100")
        if model["use_mean_embeddings"] is not True:
            raise ConfigError("Phase 2 uses mean AION embeddings")

        training = _section(
            data,
            "training",
            {
                "batch_size",
                "epochs",
                "learning_rate",
                "weight_decay",
                "gradient_clip_max_norm",
                "num_workers",
                "checkpoint_metric",
                "minimum_steps_per_epoch",
            },
            {
                "batch_size",
                "epochs",
                "learning_rate",
                "weight_decay",
                "gradient_clip_max_norm",
                "num_workers",
                "checkpoint_metric",
                "minimum_steps_per_epoch",
            },
        )
        if training["checkpoint_metric"] != "caption_to_image_recall_at_10":
            raise ConfigError("training checkpoint metric must be caption_to_image_recall_at_10")
        estimated_train_rows = int(source["sample_size"] * float(source["train_ratio"]))
        estimated_steps = (estimated_train_rows + int(training["batch_size"]) - 1) // int(
            training["batch_size"]
        )
        if estimated_steps < int(training["minimum_steps_per_epoch"]):
            raise ConfigError("training batch_size yields too few optimizer steps per epoch")

        conditions = data["conditions"]
        if not isinstance(conditions, list) or len(conditions) != 3:
            raise ConfigError("Phase 2 requires exactly three conditions")
        names = {condition.get("name") for condition in conditions if isinstance(condition, dict)}
        if names != {"R-OAI", "R-QWEN", "Q-QWEN"}:
            raise ConfigError("Phase 2 conditions must be R-OAI, R-QWEN, and Q-QWEN")
        for condition in conditions:
            if set(condition) != {"name", "text_source", "text_input_dim"}:
                raise ConfigError(f"Condition {condition.get('name')} has the wrong fields")

    return dict(data)


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    return validate_config(raw)
