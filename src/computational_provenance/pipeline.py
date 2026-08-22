"""End-to-end deterministic replay over the released evaluation rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .carrier import aggregate, detected, generate, surface_payload
from .feedforward import CausalStrictBottleneck
from .receipts import issue_execution, verify_execution
from .task import alias_state, task
from .transformer import EngineeredCausalTransformer


ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(architecture: str, split: str) -> list[dict[str, Any]]:
    path = ROOT / "data" / architecture / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_model(architecture: str, checkpoint_path: Path | None = None):
    """Load the released checkpoint or a checkpoint produced by ``scripts/train.py``."""
    path = ROOT / "checkpoints" / f"{architecture}.pt" if checkpoint_path is None else checkpoint_path
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = payload.get("step", payload.get("steps"))
    parameter_count = payload.get("parameter_count", sum(value.numel() for value in payload["state_dict"].values()))
    recorded_architecture = payload.get("architecture")
    if recorded_architecture in ("feedforward", "transformer") and recorded_architecture != architecture:
        raise ValueError("checkpoint architecture")
    if architecture == "feedforward":
        if parameter_count != 12648 or step != 5000:
            raise ValueError("feed-forward checkpoint identity")
        model = CausalStrictBottleneck()
    elif architecture == "transformer":
        if parameter_count != 546088 or step != 12000:
            raise ValueError("transformer checkpoint identity")
        model = EngineeredCausalTransformer()
    else:
        raise ValueError("architecture must be feedforward or transformer")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval(), file_sha256(path)


def _observed(output: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    return (int(output["z2_consumed"][0]), int(output["z3_selected"][0]),
            int(output["answer_logits"].argmax(-1)[0]))


@torch.no_grad()
def replay(architecture: str, split: str, *, limit: int | None = None,
           checkpoint_path: Path | None = None) -> dict[str, Any]:
    rows = load_rows(architecture, split)
    if limit is not None:
        rows = rows[:limit]
    model, checkpoint = load_model(architecture, checkpoint_path)
    receipt_key = hashlib.sha256(f"replay-receipt:{architecture}:{split}".encode()).digest()
    sampler_key = hashlib.sha256(f"replay-sampler:{architecture}:{split}".encode()).digest()
    cases = []
    for row in rows:
        inputs = row["inputs"]
        natural = task(**inputs)
        alternative = alias_state(natural)
        values = torch.tensor([[inputs[key] for key in ("a", "b", "c", "d")]], dtype=torch.int64)
        natural_output = model.run_natural(values)
        alternative_output = model.run_with_z2(values, torch.tensor([alternative["z2"]], dtype=torch.int64))
        side_results = []
        payload = surface_payload(row["prompt_text"], natural["answer"])
        for role, expected, output in (("NATURAL", natural, natural_output), ("ALTERNATIVE", alternative, alternative_output)):
            observed = _observed(output)
            computation = observed == (expected["z2"], expected["z3"], expected["answer"])
            receipts = issue_execution(receipt_key, model_domain=architecture, checkpoint_sha256=checkpoint,
                                       closure_id=row["closure_id"], role=role, z2=observed[0], z3=observed[1], answer=observed[2])
            receipt_valid = verify_execution(receipt_key, receipts)
            reports = [generate(payload, observed[0], sampler_key, row["closure_id"], index) for index in range(8)]
            carrier_scores = aggregate(reports)
            carrier_valid = detected(carrier_scores, observed[0])
            carrier_off = [generate(payload, None, sampler_key, row["closure_id"] + ":off:" + role, index) for index in range(8)]
            off_scores = aggregate(carrier_off)
            off_valid = max(off_scores) <= 4.041451884327381
            side_results.append({"role": role, "expected": expected, "observed": observed,
                                 "computation": computation, "receipt": receipt_valid,
                                 "carrier": carrier_valid, "carrier_off": off_valid,
                                 "authenticated_score": carrier_scores[observed[0]],
                                 "largest_wrong_score": max(score for index, score in enumerate(carrier_scores) if index != observed[0])})
        cases.append({"closure_id": row["closure_id"], "sides": side_results,
                      "passed": all(all(side[key] for key in ("computation", "receipt", "carrier", "carrier_off")) for side in side_results)})
    return {"architecture": architecture, "split": split, "closures": len(cases),
            "complete_successes": sum(case["passed"] for case in cases), "cases": cases}
