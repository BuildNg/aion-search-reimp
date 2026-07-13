"""Text-only Galaxy Zoo extraction for the replacement Phase 1 audit."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


ANSWER_VALUES = {
    "smooth-or-featured": {
        "smooth",
        "featured-or-disk",
        "artifact",
        "not-stated",
    },
    "how-rounded": {
        "round",
        "in-between",
        "cigar-shaped",
        "not-applicable",
        "not-stated",
    },
    "disk-edge-on": {"yes", "no", "not-applicable", "not-stated"},
    "edge-on-bulge": {
        "boxy",
        "none",
        "rounded",
        "not-applicable",
        "not-stated",
    },
    "has-spiral-arms": {"yes", "no", "not-applicable", "not-stated"},
    "spiral-winding": {
        "tight",
        "medium",
        "loose",
        "not-applicable",
        "not-stated",
    },
    "spiral-arm-count": {
        "1",
        "2",
        "3",
        "4",
        "more-than-4",
        "cant-tell",
        "not-applicable",
        "not-stated",
    },
    "bar": {"strong", "weak", "no", "not-applicable", "not-stated"},
    "bulge-size": {
        "dominant",
        "large",
        "moderate",
        "small",
        "none",
        "not-applicable",
        "not-stated",
    },
    "merging": {
        "none",
        "minor-disturbance",
        "major-disturbance",
        "merger",
        "not-applicable",
        "not-stated",
    },
}


@dataclass(frozen=True)
class MorphologyResult:
    object_id: str
    answers: Dict[str, str]
    evidence: Dict[str, Optional[str]]
    raw_response: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "answers_json": json.dumps(self.answers, sort_keys=True),
            "evidence_json": json.dumps(self.evidence, sort_keys=True),
            "raw_response": self.raw_response,
        }


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def parse_morphology_response(
    object_id: str,
    description: str,
    response: str,
) -> MorphologyResult:
    try:
        payload = json.loads(_strip_fence(response))
    except json.JSONDecodeError as error:
        raise ValueError(f"Morphology response is not valid JSON for {object_id}") from error
    if set(payload) != {"answers"} or not isinstance(payload["answers"], dict):
        raise ValueError(f"Morphology response must contain only answers for {object_id}")
    if set(payload["answers"]) != set(ANSWER_VALUES):
        raise ValueError(f"Morphology response has the wrong question fields for {object_id}")

    description_normalized = _normalize_evidence(description)
    answers: Dict[str, str] = {}
    evidence: Dict[str, Optional[str]] = {}
    for question, item in payload["answers"].items():
        if not isinstance(item, dict) or set(item) != {"value", "evidence"}:
            raise ValueError(f"{question} must contain value and evidence for {object_id}")
        value = item["value"]
        cited = item["evidence"]
        if value not in ANSWER_VALUES[question]:
            raise ValueError(f"Invalid {question} answer {value!r} for {object_id}")
        if value in {"not-stated", "not-applicable"}:
            if cited not in {None, ""}:
                raise ValueError(f"{question} must not cite evidence for {value} in {object_id}")
            cited = None
        else:
            if not isinstance(cited, str) or not cited.strip():
                raise ValueError(f"{question} requires copied evidence for {object_id}")
            if _normalize_evidence(cited) not in description_normalized:
                raise ValueError(f"{question} evidence is absent from the caption for {object_id}")
            cited = cited.strip()
        answers[question] = str(value)
        evidence[question] = cited
    return MorphologyResult(str(object_id), answers, evidence, response)


class GemmaMorphologyExtractor:
    """Gemma 4 text-only extractor; it never receives an image."""

    def __init__(
        self,
        model_path: str,
        prompt_template: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1536,
        enable_thinking: bool = False,
        processor: Any = None,
        model: Any = None,
    ) -> None:
        if enable_thinking:
            raise ValueError("The primary Phase 1 extractor must run with thinking disabled")
        if processor is None or model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype_value = getattr(torch, dtype)
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=dtype_value,
                device_map="auto",
                local_files_only=True,
            ).eval()
        self.processor = processor
        self.model = model
        self.prompt_template = prompt_template
        self.max_new_tokens = int(max_new_tokens)
        self.enable_thinking = bool(enable_thinking)

    def generate_response(self, description: str) -> str:
        import torch

        prompt = self.prompt_template.replace("{description}", description)
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        decoded = self.processor.decode(outputs[0][input_length:], skip_special_tokens=False)
        if hasattr(self.processor, "parse_response"):
            decoded = _response_text(self.processor.parse_response(decoded))
        return decoded.strip()

    def extract(self, object_id: str, description: str) -> MorphologyResult:
        response = self.generate_response(description)
        return parse_morphology_response(object_id, description, response)


def _response_text(parsed: Any) -> str:
    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, Mapping):
        for key in ("content", "text", "response"):
            if isinstance(parsed.get(key), str):
                return str(parsed[key])
    if isinstance(parsed, list):
        for item in reversed(parsed):
            if isinstance(item, Mapping):
                for key in ("content", "text", "response"):
                    if isinstance(item.get(key), str):
                        return str(item[key])
            elif isinstance(item, str):
                return item
    for attribute in ("content", "text", "response"):
        value = getattr(parsed, attribute, None)
        if isinstance(value, str):
            return value
    raise ValueError(f"Unsupported processor.parse_response result: {type(parsed).__name__}")


def append_morphology_results(
    extractor: Any,
    caption_rows: Iterable[Mapping[str, Any]],
    output_jsonl: Path,
    error_jsonl: Optional[Path] = None,
    max_error_rate: float = 0.0,
) -> Dict[str, Any]:
    if not 0.0 <= max_error_rate < 1.0:
        raise ValueError("max_error_rate must be in [0, 1)")
    rows = list(caption_rows)
    ids = [str(row["object_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Morphology input contains duplicate object_id values")
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_object_ids(output_jsonl)
    failed = _read_object_ids(Path(error_jsonl)) if error_jsonl is not None else set()
    if completed & failed:
        raise ValueError("Morphology outputs and errors overlap")
    unexpected = (completed | failed) - set(ids)
    if unexpected:
        raise ValueError(f"Unexpected morphology object_id={sorted(unexpected)[0]}")
    error_budget = math.floor(len(rows) * max_error_rate)
    if len(failed) > error_budget:
        raise RuntimeError("Existing morphology errors exceed the configured cap")

    error_handle = None
    if error_jsonl is not None:
        error_path = Path(error_jsonl)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_handle = error_path.open("a", encoding="utf-8")
    try:
        with output_jsonl.open("a", encoding="utf-8") as output_handle:
            for row in rows:
                object_id = str(row["object_id"])
                if object_id in completed or object_id in failed:
                    continue
                description = str(row["description"])
                raw_response: Optional[str] = None
                try:
                    raw_response = extractor.generate_response(description)
                    result = parse_morphology_response(
                        object_id, description, raw_response
                    )
                except Exception as error:
                    if error_handle is None:
                        raise
                    error_handle.write(
                        json.dumps(
                            {
                                "object_id": object_id,
                                "stage": "parse" if raw_response is not None else "generation",
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "raw_response": raw_response,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    error_handle.flush()
                    failed.add(object_id)
                    if len(failed) > error_budget:
                        raise RuntimeError(
                            f"Morphology error cap exceeded: {len(failed)}/{len(rows)}"
                        ) from error
                    continue
                output_handle.write(json.dumps(result.as_record(), sort_keys=True) + "\n")
                output_handle.flush()
                completed.add(object_id)
    finally:
        if error_handle is not None:
            error_handle.close()

    attempted = len(completed) + len(failed)
    return {
        "input_rows": len(rows),
        "completed_rows": len(completed),
        "error_rows": len(failed),
        "attempted_rows": attempted,
        "pending_rows": len(rows) - attempted,
        "error_rate": 0.0 if attempted == 0 else len(failed) / attempted,
        "max_error_rate": max_error_rate,
    }


def _read_object_ids(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {
        str(json.loads(line)["object_id"])
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def calibration_metrics(extractor: Any, calibration_path: Path) -> Dict[str, Any]:
    records = [
        json.loads(line)
        for line in Path(calibration_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("Extractor calibration set is empty")
    answer_total = 0
    answer_correct = 0
    row_exact = 0
    row_records = []
    for record in records:
        result = extractor.extract(str(record["object_id"]), str(record["description"]))
        expected = {str(key): str(value) for key, value in record["expected"].items()}
        correctness = {key: result.answers.get(key) == value for key, value in expected.items()}
        answer_total += len(correctness)
        answer_correct += sum(correctness.values())
        exact = all(correctness.values())
        row_exact += int(exact)
        row_records.append(
            {
                "object_id": str(record["object_id"]),
                "exact": exact,
                "correctness": correctness,
                "answers": result.answers,
            }
        )
    return {
        "rows": len(records),
        "row_exact_accuracy": row_exact / len(records),
        "answers": answer_total,
        "answer_accuracy": answer_correct / answer_total,
        "records": row_records,
    }
