import torch

from spec_probes.specformer_model import SpecFormer


def _tiny_model() -> SpecFormer:
    # slice_section_length=20, slice_overlap=10 (defaults) -> a length-40
    # input yields exactly 3 full sections plus the prepended stats row.
    return SpecFormer(input_dim=22, embed_dim=16, num_layers=2, num_heads=2, max_len=10, dropout=0.0).eval()


def test_specformer_forward_returns_reconstructions_and_embedding() -> None:
    model = _tiny_model()
    x = torch.randn(3, 40, 1)
    with torch.no_grad():
        output = model(x)
    assert set(output.keys()) == {"reconstructions", "embedding"}
    assert output["embedding"].shape[0] == 3
    assert output["embedding"].shape[2] == 16
    assert output["reconstructions"].shape == x.shape[:1] + output["reconstructions"].shape[1:]


def test_specformer_embedding_is_deterministic_in_eval_mode() -> None:
    model = _tiny_model()
    x = torch.randn(2, 40, 1)
    with torch.no_grad():
        first = model(x)["embedding"]
        second = model(x)["embedding"]
    torch.testing.assert_close(first, second)


def test_specformer_state_dict_round_trips() -> None:
    """The vendored module must accept the upstream checkpoint's state dict
    shape (same parameter names/shapes as astroclip/models/specformer.py)."""
    model = _tiny_model()
    state = model.state_dict()
    clone = _tiny_model()
    clone.load_state_dict(state)  # must not raise
    expected_keys = {"data_embed", "position_embed", "blocks", "final_layernorm", "head"}
    top_level_prefixes = {key.split(".")[0] for key in state.keys()}
    assert expected_keys <= top_level_prefixes


def test_specformer_rejects_sequence_longer_than_max_len() -> None:
    model = SpecFormer(input_dim=22, embed_dim=16, num_layers=1, num_heads=2, max_len=1, dropout=0.0).eval()
    x = torch.randn(1, 40, 1)
    try:
        with torch.no_grad():
            model(x)
        raised = False
    except ValueError:
        raised = True
    assert raised
