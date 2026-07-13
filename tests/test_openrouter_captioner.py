from pathlib import Path
from types import SimpleNamespace

import pytest

from aion_reimp.captioning import OpenRouterCaptioner, QwenCaptioner


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        usage = SimpleNamespace(
            model_dump=lambda: {
                "prompt_tokens": 500,
                "completion_tokens": 150,
                "total_tokens": 650,
                "cost": 0.00044,
            }
        )
        return SimpleNamespace(
            id="request-1",
            model="openai/gpt-4.1-mini-2025-04-14",
            provider="OpenAI",
            usage=usage,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A free-form galaxy description.")
                )
            ],
        )


def test_openrouter_captioner_pins_provider_without_structuring_output(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"small-fixture")
    completions = _FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    captioner = OpenRouterCaptioner(
        model_id="openai/gpt-4.1-mini-2025-04-14",
        prompt="Describe the galaxy.",
        api_key="fixture",
        client=client,
    )
    response, usage = captioner.generate_response_with_metadata(image_path)

    assert response == "A free-form galaxy description."
    assert usage["reported_cost_usd"] == 0.00044
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["extra_body"]["provider"] == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
    }
    assert "response_format" not in completions.kwargs
    content = completions.kwargs["messages"][0]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]


def test_qwen_uses_the_same_prompt_then_image_order(tmp_path: Path) -> None:
    class _CaptureProcessor:
        messages = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            raise RuntimeError("captured")

    captioner = QwenCaptioner.__new__(QwenCaptioner)
    captioner.processor = _CaptureProcessor()
    captioner.prompt = "Describe the galaxy."
    with pytest.raises(RuntimeError, match="captured"):
        captioner.generate_response(tmp_path / "image.png")
    content = captioner.processor.messages[0]["content"]
    assert [item["type"] for item in content] == ["text", "image"]
