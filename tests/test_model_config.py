import pytest

from aion_reimp.model import ModelConfig


VALID = {
    "image_input_dim": 768,
    "text_input_dim": 3072,
    "embedding_dim": 1024,
    "image_hidden_dim": 768,
    "text_hidden_dim": 1024,
    "dropout": 0.1,
    "use_mean_embeddings": True,
}


def test_released_config_requires_every_key() -> None:
    values = dict(VALID)
    values.pop("image_hidden_dim")
    with pytest.raises(ValueError, match="missing required keys"):
        ModelConfig.from_mapping(values)


def test_cross_attention_scope_is_rejected() -> None:
    values = dict(VALID)
    values["use_mean_embeddings"] = False
    with pytest.raises(ValueError, match="use_mean_embeddings=true"):
        ModelConfig.from_mapping(values)
