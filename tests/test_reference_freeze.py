import json
import sys
from types import SimpleNamespace

import pandas as pd

from aion_reimp.reference import freeze_openai_queries


class _FakeEmbeddings:
    def create(self, **kwargs):
        assert kwargs["model"] == "openai/text-embedding-3-large"
        assert kwargs["extra_body"]["provider"]["allow_fallbacks"] is False
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0] * 3072),
                SimpleNamespace(index=0, embedding=[1.0] * 3072),
            ],
            model="openai/text-embedding-3-large",
            id="response-1",
            usage=None,
        )


class _FakeOpenAI:
    def __init__(self, api_key, base_url):
        assert api_key == "secret"
        assert base_url == "https://openrouter.ai/api/v1"
        self.embeddings = _FakeEmbeddings()


def test_freeze_sorts_response_and_records_gateway(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPEN_ROUTER_KEY", "secret")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_FakeOpenAI))
    output = tmp_path / "queries.parquet"
    rows = [
        {"object_id": "q0", "text": "first"},
        {"object_id": "q1", "text": "second"},
    ]

    freeze_openai_queries(rows, output)

    frame = pd.read_parquet(output)
    assert frame.iloc[0].embedding[0] == 1.0
    assert frame.iloc[1].embedding[0] == 2.0
    metadata = json.loads(output.with_suffix(".parquet.meta.json").read_text(encoding="utf-8"))
    assert metadata["gateway"] == "openrouter"
    assert metadata["provider_policy"]["allow_fallbacks"] is False
