from pathlib import Path

import torch

from computational_provenance.feedforward import CausalStrictBottleneck
from computational_provenance.task import alias_state, task
from computational_provenance.transformer import EngineeredCausalTransformer, architecture_record


ROOT = Path(__file__).resolve().parents[1]


def load_model(architecture: str):
    path = ROOT / "checkpoints" / f"{architecture}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CausalStrictBottleneck() if architecture == "feedforward" else EngineeredCausalTransformer()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), __import__("hashlib").sha256(path.read_bytes()).hexdigest()


def test_registered_alias_preserves_answer() -> None:
    natural = task(4, 15, 1, 1)
    alternative = alias_state(natural)
    assert natural == {"z1": 3, "z2": 4, "z3": 5, "answer": 5}
    assert alternative == {"z1": 3, "z2": 12, "z3": 13, "answer": 5}


def test_parameter_counts() -> None:
    assert sum(parameter.numel() for parameter in CausalStrictBottleneck().parameters()) == 12648
    assert architecture_record()["parameter_count"] == 546088
    assert architecture_record()["shared_stage_parameter_objects"] == 0


def test_final_checkpoints_load() -> None:
    for architecture in ("feedforward", "transformer"):
        model, digest = load_model(architecture)
        assert len(digest) == 64
        assert not model.training
        assert (ROOT / "checkpoints" / f"{architecture}.pt").is_file()


def test_intervention_changes_consumed_transformer_state() -> None:
    model, _ = load_model("transformer")
    values = torch.tensor([[4, 15, 1, 1]], dtype=torch.int64)
    first = model.run_with_z2(values, torch.tensor([4], dtype=torch.int64))
    second = model.run_with_z2(values, torch.tensor([12], dtype=torch.int64))
    assert int(first["z2_consumed"][0]) == 4
    assert int(second["z2_consumed"][0]) == 12
    assert int(first["z3_selected"][0]) != int(second["z3_selected"][0])
