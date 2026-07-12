"""Deterministic Qwen3-VL caption generation and schema parsing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


ANSWER_VALUES = {
    "smooth-or-featured": {"smooth", "featured-or-disk", "artifact", "uncertain"},
    "how-rounded": {"round", "in-between", "cigar-shaped", "not-applicable", "uncertain"},
    "disk-edge-on": {"yes", "no", "not-applicable", "uncertain"},
    "edge-on-bulge": {"boxy", "none", "rounded", "not-applicable", "uncertain"},
    "has-spiral-arms": {"yes", "no", "not-applicable", "uncertain"},
    "spiral-winding": {"tight", "medium", "loose", "not-applicable", "uncertain"},
    "spiral-arm-count": {"1", "2", "3", "4", "more-than-4", "cant-tell", "not-applicable", "uncertain"},
    "bar": {"strong", "weak", "no", "not-applicable", "uncertain"},
    "bulge-size": {"dominant", "large", "moderate", "small", "none", "not-applicable", "uncertain"},
    "merging": {"none", "minor-disturbance", "major-disturbance", "merger", "uncertain"},
}


@dataclass(frozen=True)
class CaptionResult:
    object_id: str
    summary: str
    answers: Dict[str, str]
    raw_response: str

    def as_record(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "summary": self.summary,
            "answers_json": json.dumps(self.answers, sort_keys=True),
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


def parse_caption_response(object_id: str, response: str) -> CaptionResult:
    try:
        payload = json.loads(_strip_fence(response))
    except json.JSONDecodeError as error:
        raise ValueError(f"Caption response is not valid JSON for {object_id}") from error
    if set(payload) != {"summary", "answers"}:
        raise ValueError(f"Caption response must contain only summary and answers for {object_id}")
    summary = payload["summary"]
    answers = payload["answers"]
    if not isinstance(summary, str) or not summary.strip() or "\n" in summary.strip():
        raise ValueError(f"Summary must be one non-empty line for {object_id}")
    if not isinstance(answers, dict) or set(answers) != set(ANSWER_VALUES):
        raise ValueError(f"Caption answers have the wrong fields for {object_id}")
    for question, value in answers.items():
        if value not in ANSWER_VALUES[question]:
            raise ValueError(f"Invalid {question} answer {value!r} for {object_id}")
    return CaptionResult(str(object_id), summary.strip(), dict(answers), response)


class QwenCaptioner:
    """Cluster-only model wrapper. Construction loads model weights."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        prompt: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype_value = getattr(torch, dtype)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype_value,
            device_map="auto",
        ).eval()
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)

    def generate(self, object_id: str, image_path: Path) -> CaptionResult:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(Path(image_path).resolve())},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return parse_caption_response(object_id, response)


def append_caption_results(
    captioner: QwenCaptioner,
    rows: Iterable[Mapping[str, Any]],
    output_jsonl: Path,
) -> None:
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["object_id"]))
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for row in rows:
            object_id = str(row["object_id"])
            if object_id in completed:
                continue
            result = captioner.generate(object_id, Path(row["image_path"]))
            handle.write(json.dumps(result.as_record(), sort_keys=True) + "\n")
            handle.flush()
