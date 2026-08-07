"""Dynamic-width representation, controls, and exact 78,400-call registry."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from ..artifacts import tensor_sha256
from ..model import declared_iit_basis
from ..remediation.strict import object_sha256, sha256_file, write_json, write_jsonl
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .runtime import load_checkpoint


def representation_manifest() -> dict:
    value = {
        "schema": "exp33ae.dev_only_reliability_representation.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(), "target": "z2",
        "layer": 2, "position": 14, "rank": 2, "strength": 1.0,
        "basis_convention": "B=[e0,e1] in the model residual width",
        "decoder": {"kind": "fixed_registered_coordinate_argmax", "coordinates": [0, 1], "classes": 2},
        "state_codes": {"0": [1.0, 0.0], "1": [0.0, 1.0]},
        "primary_method": "declared_iit_alignment", "secondary_methods": [],
        "das_and_probes_adjudicate": False,
    }
    value["representation_sha256"] = object_sha256(value); return value


def _random_basis(width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(width, 2, generator=generator, dtype=torch.float64), mode="reduced").Q


def control_manifest(entropy, training: dict) -> dict:
    width = training["model_config"]["d_model"]
    config = type("Config", (), {"d_model": width})()
    primary = declared_iit_basis(config).float(); primary64 = primary.double()
    model_order = training["model_order"]; controls = []
    for index in range(4):
        seed = int.from_bytes(entropy.derive("random-direction", model_order * 4 + index, 8), "big") & ((1 << 63) - 1)
        basis = _random_basis(width, seed).float()
        controls.append({"control_id": f"random_direction_{index}", "family": "random_direction", "index": index,
                         "layer": 2, "position": 14, "rank": 2, "strength": 1.0,
                         "basis": basis.tolist(), "basis_sha256": tensor_sha256(basis)})
    for index in range(4):
        seed = int.from_bytes(entropy.derive("orthogonal-direction", model_order * 4 + index, 8), "big") & ((1 << 63) - 1)
        raw = _random_basis(width, seed); residual = raw - primary64 @ (primary64.T @ raw)
        basis = torch.linalg.qr(residual, mode="reduced").Q.float()
        if float((primary.T @ basis).abs().max()) > 1e-5: raise RuntimeError("orthogonal basis failure")
        controls.append({"control_id": f"orthogonal_direction_{index}", "family": "orthogonal_direction", "index": index,
                         "layer": 2, "position": 14, "rank": 2, "strength": 1.0,
                         "basis": basis.tolist(), "basis_sha256": tensor_sha256(basis)})
    for index, position in enumerate((12, 13, 15)):
        controls.append({"control_id": f"alternate_location_{index}", "family": "alternate_location", "index": index,
                         "layer": 2, "position": position, "rank": 2, "strength": 1.0,
                         "basis": primary.tolist(), "basis_sha256": tensor_sha256(primary)})
    controls.append({"control_id": "alternate_early_layer_0", "family": "alternate_early_layer", "index": 0,
                     "layer": 0, "position": 14, "rank": 2, "strength": 1.0,
                     "basis": primary.tolist(), "basis_sha256": tensor_sha256(primary)})
    value = {
        "schema": "exp33ae.dev_only_reliability_controls.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "protocol_sha256": protocol_hash(), "model_order": model_order,
        "recipe_id": training["recipe_id"], "recipe_order": training["recipe_order"],
        "seed_order": training["seed_order"], "model_seed_identity": training["model_seed_identity"],
        "residual_width": width, "primary_basis_sha256": tensor_sha256(primary), "controls": controls,
        "norm_tolerance_absolute": 1e-6, "norm_tolerance_relative": 1e-5,
    }
    value["control_manifest_sha256"] = object_sha256(value); return value


def _primary_norms(model, pairs, examples) -> dict[str, float]:
    basis = declared_iit_basis(model.config); values = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), 256):
            batch = pairs[start:start+256]; bases = [examples[p["base_id"]] for p in batch]; sources = [examples[p["source_id"]] for p in batch]
            bt = torch.tensor([r.tokens for r in bases]); btab = torch.tensor([r.operation_tables for r in bases], dtype=torch.float32)
            st = torch.tensor([r.tokens for r in sources]); stab = torch.tensor([r.operation_tables for r in sources], dtype=torch.float32)
            _, acts, _ = model(bt, btab, return_activations=True); src = model.source_activation(st, stab, 2, 14)
            norms = (((src - acts[2][:, 14, :]) @ basis) @ basis.T).norm(dim=-1)
            for plan, norm in zip(batch, norms.tolist()): values[plan["pair_id"]] = float(norm)
    return values


def build_registry(output: Path, entropy, allocation: dict, pairs: list[dict], allocation_manifest: dict,
                   training: list[dict], representation: dict, source: dict, dependency: dict) -> dict:
    examples = allocation["examples"]; by_id = {r.example_id: r for r in examples}
    clean = [r for r in examples if r.partition in {"clean_target_axis", "clean_parallel_axis"}]
    A = [p for p in pairs if p["stratum"] == "A_answer_changing"]
    registry = []; ordinal = 0
    for train in training:
        expected = {"recipe_id": train["recipe_id"], "recipe_order": train["recipe_order"],
                    "seed_order": train["seed_order"], "model_seed_identity": train["model_seed_identity"],
                    "source_manifest_sha256": source["manifest_sha256"],
                    "dependency_manifest_sha256": dependency["manifest_sha256"],
                    "tensor_state_sha256": train["final_tensor_state_sha256"],
                    "checkpoint_file_sha256": train["checkpoint_file_sha256"]}
        model, _ = load_checkpoint(output / train["checkpoint_relative_path"], expected)
        norms = _primary_norms(model, pairs, by_id)
        controls = control_manifest(entropy, train)
        control_path = output / "controls" / train["recipe_id"] / f"seed-{train['seed_order']}.json"
        write_json(control_path, controls)
        basis_primary = declared_iit_basis(model.config); primary_hash = tensor_sha256(basis_primary)

        def append(base, source_example, pair, method: str, control: dict | None) -> None:
            nonlocal ordinal
            if pair is None:
                stratum = "clean_task_decode"; expected_values = {"z2": base.z2, "z3": base.z3, "n3": base.n3, "answer": base.answer}
                pair_id = f"devonly-rel-clean-{train['model_order']}-{base.example_id}"; collision = False; same_answer = True; no_op = True
                layer, position, rank, strength, control_id, basis_hash, expected_norm = 2, 14, 2, 0.0, None, primary_hash, 0.0
            else:
                stratum = pair["stratum"]; expected_values = {k: pair[f"expected_{k}"] for k in ("z2", "z3", "n3", "answer")}
                pair_id = pair["pair_id"]; collision = pair["collision"]; same_answer = pair["same_observable_answer"]; no_op = pair["no_op"]
                expected_norm = norms[pair_id]
                if control is None:
                    layer, position, rank, strength, control_id, basis_hash = 2, 14, 2, 1.0, None, primary_hash
                else:
                    layer, position, rank, strength, control_id, basis_hash = control["layer"], control["position"], control["rank"], control["strength"], control["control_id"], control["basis_sha256"]
            row = {
                "schema": "exp33ae.dev_only_reliability_registry_row.v1", "namespace": DEV_NAMESPACE,
                "claim_bearing": False, "stage_id": entropy.stage_id, "protocol_sha256": protocol_hash(),
                "source_manifest_sha256": source["manifest_sha256"], "dependency_manifest_sha256": dependency["manifest_sha256"],
                "allocation_sha256": allocation_manifest["allocation_sha256"], "representation_sha256": representation["representation_sha256"],
                "control_manifest_sha256": controls["control_manifest_sha256"], "training_manifest_sha256": train["scientific_manifest_sha256"],
                "checkpoint_relative_path": train["checkpoint_relative_path"], "checkpoint_file_sha256": train["checkpoint_file_sha256"],
                "checkpoint_tensor_sha256": train["final_tensor_state_sha256"], "model_order": train["model_order"],
                "recipe_id": train["recipe_id"], "recipe_order": train["recipe_order"], "seed_order": train["seed_order"],
                "model_seed_identity": train["model_seed_identity"], "order": ordinal, "method": method, "control_id": control_id,
                "programme_id": base.programme_id, "causal_signature": base.causal_signature, "axis": base.axis,
                "base_id": base.example_id, "source_id": source_example.example_id, "pair_id": pair_id, "pair_stratum": stratum,
                "eligible": True, "collision": collision, "same_observable_answer": same_answer, "no_op": no_op,
                "base_z2": base.z2, "base_z3": base.z3, "base_n3": base.n3, "base_answer": base.answer,
                "source_z2": source_example.z2, "source_z3": source_example.z3, "source_n3": source_example.n3, "source_answer": source_example.answer,
                "expected_z2": expected_values["z2"], "expected_z3": expected_values["z3"], "expected_n3": expected_values["n3"], "expected_answer": expected_values["answer"],
                "basis_sha256": basis_hash, "decoder_sha256": representation["representation_sha256"],
                "layer": layer, "position": position, "rank": rank, "strength": strength,
                "expected_applied_norm": expected_norm, "norm_tolerance_absolute": 1e-6, "norm_tolerance_relative": 1e-5,
                "call_id": "devonly-relcall-" + entropy.derive("call-id", ordinal, 16).hex(),
                "nonce": "devonly-relnonce-" + entropy.derive("nonce", ordinal, 16).hex(),
                "planned_attempts": 1, "retry_policy": "resume-first-missing-no-reexecution",
            }
            row["registry_row_sha256"] = object_sha256(row); registry.append(row); ordinal += 1

        for example in clean: append(example, example, None, "clean_task_decode", None)
        for pair in pairs: append(by_id[pair["base_id"]], by_id[pair["source_id"]], pair, "declared_iit_alignment", None)
        for control in controls["controls"]:
            for pair in A: append(by_id[pair["base_id"]], by_id[pair["source_id"]], pair, control["family"], control)
    if len(registry) != 78400: raise RuntimeError(f"registry count {len(registry)}")
    payload = object_sha256(registry)
    for row in registry: row["registry_payload_commitment"] = payload
    path = output / "registry" / "registry.jsonl"; write_jsonl(path, registry)
    manifest = {
        "schema": "exp33ae.dev_only_reliability_registry_manifest.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "stage_id": entropy.stage_id, "protocol_sha256": protocol_hash(),
        "model_seed_identities": list(entropy.model_seeds()), "recipe_order": [r["id"] for r in load_protocol()["recipes"]],
        "row_count": len(registry), "rows_per_model": 3920, "rows_per_recipe": 19600, "models": 20,
        "payload_commitment": payload, "registry_file_sha256": sha256_file(path),
        "method_counts": dict(sorted(Counter(r["method"] for r in registry).items())),
        "generated_before_evaluation": True, "confirmation_registry_materialized": False,
    }
    manifest["registry_manifest_sha256"] = object_sha256(manifest); write_json(output / "registry" / "manifest.json", manifest)
    return {"rows": registry, "manifest": manifest}
