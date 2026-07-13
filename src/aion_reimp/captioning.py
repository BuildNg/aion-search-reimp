"""Free-form galaxy descriptions from the two Phase 1 vision models."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


@dataclass(frozen=True)
class CaptionResult:
    object_id: str
    description: str
    raw_response: str
    word_count: int

    def as_record(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "description": self.description,
            "raw_response": self.raw_response,
            "word_count": self.word_count,
        }


def parse_caption_response(object_id: str, response: str) -> CaptionResult:
    """Validate a non-empty response while preserving all returned text."""
    if not isinstance(response, str) or not response.strip():
        raise ValueError(f"Caption response is empty for {object_id}")
    description = response.strip()
    word_count = len(description.split())
    return CaptionResult(str(object_id), description, response, word_count)


class OpenRouterCaptioner:
    """Pinned GPT-4.1-mini vision reference accessed locally through OpenRouter."""

    def __init__(
        self,
        model_id: str,
        prompt: str,
        api_key: str,
        provider: str = "OpenAI",
        base_url: str = "https://openrouter.ai/api/v1",
        max_output_tokens: int = 1024,
        temperature: float = 0.0,
        image_detail: str = "low",
        client: Any = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is empty")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be low, high, or auto")
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=3)
        self.client = client
        self.model_id = model_id
        self.provider = provider
        self.prompt = prompt
        self.max_output_tokens = int(max_output_tokens)
        self.temperature = float(temperature)
        self.image_detail = image_detail

    @staticmethod
    def _image_data_url(image_path: Path) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def generate_response_with_metadata(self, image_path: Path) -> tuple[str, Dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self._image_data_url(image_path),
                                "detail": self.image_detail,
                            },
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_completion_tokens=self.max_output_tokens,
            extra_body={
                "provider": {"order": [self.provider], "allow_fallbacks": False},
                "usage": {"include": True},
            },
        )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("OpenRouter returned an empty caption response")
        usage = response.usage.model_dump() if response.usage is not None else {}
        metadata = {
            "request_id": response.id,
            "returned_model": response.model,
            "provider": getattr(response, "provider", None),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "reported_cost_usd": usage.get("cost"),
        }
        return content, metadata


class QwenCaptioner:
    """Cluster-only Qwen3-VL wrapper. Construction loads model weights."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        prompt: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
    ) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        import torch

        dtype_value = getattr(torch, dtype)
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            revision=revision,
            dtype=dtype_value,
            device_map="auto",
        ).eval()
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)

    def generate_response(self, image_path: Path) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image", "image": str(Path(image_path).resolve())},
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
        with self._torch.inference_mode():
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
        return response


def append_caption_results(
    captioner: Any,
    rows: Iterable[Mapping[str, Any]],
    output_jsonl: Path,
    error_jsonl: Optional[Path] = None,
    max_error_rate: float = 0.0,
) -> Dict[str, Any]:
    """Generate resumable object-keyed descriptions with an explicit error budget."""
    if not 0.0 <= max_error_rate < 1.0:
        raise ValueError("max_error_rate must be in [0, 1)")
    row_list = list(rows)
    object_ids = [str(row["object_id"]) for row in row_list]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Caption input contains duplicate object_id values")

    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    completed = _read_object_ids(output_jsonl)
    failed = _read_object_ids(Path(error_jsonl)) if error_jsonl is not None else set()
    _validate_resume_sets(object_ids, completed, failed)

    error_budget = math.floor(len(row_list) * max_error_rate)
    if len(failed) > error_budget:
        raise RuntimeError(
            f"Existing caption errors exceed cap: {len(failed)}/{len(row_list)} "
            f"> {max_error_rate:.3%}"
        )

    error_handle = None
    if error_jsonl is not None:
        error_path = Path(error_jsonl)
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_handle = error_path.open("a", encoding="utf-8")
    try:
        with output_jsonl.open("a", encoding="utf-8") as handle:
            for row in row_list:
                object_id = str(row["object_id"])
                if object_id in completed or object_id in failed:
                    continue
                image_path = Path(row["image_path"])
                raw_response: Optional[str] = None
                try:
                    raw_response = captioner.generate_response(image_path)
                    result = parse_caption_response(object_id, raw_response)
                except Exception as error:
                    if error_handle is None:
                        raise
                    record = {
                        "object_id": object_id,
                        "image_path": str(image_path),
                        "stage": "validation" if raw_response is not None else "generation",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "raw_response": raw_response,
                    }
                    error_handle.write(json.dumps(record, sort_keys=True) + "\n")
                    error_handle.flush()
                    failed.add(object_id)
                    if len(failed) > error_budget:
                        raise RuntimeError(
                            f"Caption error cap exceeded: {len(failed)}/{len(row_list)} "
                            f"> {max_error_rate:.3%}"
                        ) from error
                    continue
                handle.write(json.dumps(result.as_record(), sort_keys=True) + "\n")
                handle.flush()
                completed.add(object_id)
    finally:
        if error_handle is not None:
            error_handle.close()

    attempted = len(completed) + len(failed)
    return {
        "input_rows": len(row_list),
        "completed_rows": len(completed),
        "error_rows": len(failed),
        "attempted_rows": attempted,
        "pending_rows": len(row_list) - attempted,
        "error_rate": 0.0 if attempted == 0 else len(failed) / attempted,
        "max_error_rate": max_error_rate,
        "error_budget": error_budget,
    }


def _read_object_ids(path: Path) -> set[str]:
    if not Path(path).exists():
        return set()
    return {
        str(json.loads(line)["object_id"])
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _validate_resume_sets(
    object_ids: Iterable[str], completed: set[str], failed: set[str]
) -> None:
    overlap = completed & failed
    if overlap:
        raise ValueError(f"Caption outputs and errors overlap for object_id={sorted(overlap)[0]}")
    unexpected = (completed | failed) - set(object_ids)
    if unexpected:
        raise ValueError(
            f"Caption artifacts contain an unexpected object_id={sorted(unexpected)[0]}"
        )
