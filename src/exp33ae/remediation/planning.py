"""Frozen representation, controls, and exact pre-call registry construction."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

import torch

from ..artifacts import tensor_sha256
from ..model import TinyProgrammeTransformer
from ..preconfirmation.allocator import causal_signature
from .entropy import DevMasterEntropy
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .runtime import load_checkpoint
from .strict import object_sha256, sha256_file, write_json, write_jsonl
from .training import primary_basis


def representation_manifest() -> dict:
    basis = primary_basis(torch.float32)
    value = {
        "schema": "exp33ae.dev_only_registered_representation.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "protocol_sha256": protocol_hash(),
        "target": "z2",
        "layer": 2,
        "position": 14,
        "rank": 2,
        "strength": 1.0,
        "basis": basis.tolist(),
        "basis_sha256": tensor_sha256(basis),
        "decoder": {"kind": "registered_coordinate_argmax", "classes": 2, "coordinates": [0, 1]},
        "state_codes": {"0": [1.0, 0.0], "1": [0.0, 1.0]},
        "secondary_methods": [],
        "adjudicating_decoder": "argmax(B^T h)",
    }
    value["representation_sha256"] = object_sha256(value)
    return value


def _basis(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(64, 2, generator=generator, dtype=torch.float64), mode="reduced").Q


def control_manifest(entropy: DevMasterEntropy, model_order: int) -> dict:
    primary = primary_basis(torch.float32)
    primary_double = primary.to(torch.float64)
    controls = []
    for index in range(4):
        seed = entropy.stream_seed("random-direction", model_order * 4 + index)
        basis = _basis(seed).float()
        controls.append({"control_id": f"random_direction_{index}", "family": "random_direction", "index": index, "layer": 2, "position": 14, "rank": 2, "strength": 1.0, "basis": basis.tolist(), "basis_sha256": tensor_sha256(basis)})
    for index in range(4):
        seed = entropy.stream_seed("orthogonal-direction", model_order * 4 + index)
        raw = _basis(seed)
        residual = raw - primary_double @ (primary_double.T @ raw)
        basis = torch.linalg.qr(residual, mode="reduced").Q.float()
        if float((primary.T @ basis).abs().max()) > 1e-10:
            raise RuntimeError("orthogonal control construction failed")
        controls.append({"control_id": f"orthogonal_direction_{index}", "family": "orthogonal_direction", "index": index, "layer": 2, "position": 14, "rank": 2, "strength": 1.0, "basis": basis.tolist(), "basis_sha256": tensor_sha256(basis)})
    for index, position in enumerate((12, 13, 15)):
        controls.append({"control_id": f"alternate_location_{index}", "family": "alternate_location", "index": index, "layer": 2, "position": position, "rank": 2, "strength": 1.0, "basis": primary.tolist(), "basis_sha256": tensor_sha256(primary)})
    controls.append({"control_id": "alternate_early_layer_0", "family": "alternate_early_layer", "index": 0, "layer": 0, "position": 14, "rank": 2, "strength": 1.0, "basis": primary.tolist(), "basis_sha256": tensor_sha256(primary)})
    value = {
        "schema": "exp33ae.dev_only_control_manifest.v2",
        "namespace": DEV_NAMESPACE,
        "claim_bearing": False,
        "stage_id": entropy.stage_id,
        "protocol_sha256": protocol_hash(),
        "model_order": model_order,
        "model_seed_identity": entropy.model_seeds()[model_order],
        "controls": controls,
        "norm_tolerance_absolute": 1e-6,
        "norm_tolerance_relative": 1e-5,
    }
    value["control_manifest_sha256"] = object_sha256(value)
    return value


def _model_primary_norms(model, pairs, by_id) -> dict[str, float]:
    basis = primary_basis()
    norms = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pairs), 256):
            batch = pairs[start:start+256]
            bases = [by_id[row["base_id"]] for row in batch]
            sources = [by_id[row["source_id"]] for row in batch]
            bt = torch.tensor([row.tokens for row in bases], dtype=torch.long)
            btab = torch.tensor([row.operation_tables for row in bases], dtype=torch.float32)
            st = torch.tensor([row.tokens for row in sources], dtype=torch.long)
            stab = torch.tensor([row.operation_tables for row in sources], dtype=torch.float32)
            _, acts, _ = model(bt, btab, return_activations=True)
            source = model.source_activation(st, stab, 2, 14)
            delta = (source - acts[2][:, 14, :]) @ basis @ basis.T
            for plan, value in zip(batch, delta.norm(dim=-1).tolist()):
                norms[plan["pair_id"]] = float(value)
    return norms


def _row_hash(row: dict) -> str:
    copy = dict(row)
    copy.pop("registry_row_sha256", None)
    copy.pop("registry_payload_commitment", None)
    return object_sha256(copy)


def build_registry(
    output: Path,
    entropy: DevMasterEntropy,
    allocation: dict,
    pairs: list[dict],
    allocation_manifest: dict,
    training_manifests: list[dict],
    representation: dict,
    operative_manifest: dict,
    environment: dict,
) -> dict:
    examples = allocation["examples"]
    clean = [row for row in examples if row.partition in {"clean_target_axis", "clean_parallel_axis"}]
    by_id = {row.example_id: row for row in examples}
    A = [row for row in pairs if row["stratum"] == "A_answer_changing"]
    registry = []
    ordinal = 0
    for model_order, training in enumerate(training_manifests):
        checkpoint_path = output / training["checkpoint_relative_path"]
        model, _ = load_checkpoint(checkpoint_path, {
            "model_order": model_order,
            "model_seed_identity": training["model_seed_identity"],
            "operative_manifest_sha256": operative_manifest["manifest_sha256"],
            "environment_sha256": environment["manifest_sha256"],
            "tensor_state_sha256": training["final_tensor_state_sha256"],
            "checkpoint_file_sha256": training["checkpoint_file_sha256"],
        })
        norms = _model_primary_norms(model, pairs, by_id)
        control_path = output / "controls" / f"model-{model_order}.json"
        controls = control_manifest(entropy, model_order)
        write_json(control_path, controls)
        training_path = output / "training" / f"model-{model_order}.json"

        def append(example, pair, method: str, control: dict | None):
            nonlocal ordinal
            clean_call = pair is None
            if clean_call:
                base = source = example
                stratum = "clean_task_decode"
                expected = {"z2": base.z2, "z3": base.z3, "n3": base.n3, "answer": base.answer}
                pair_id = "devonly-clean-" + sha256(f"{model_order}:{base.example_id}".encode()).hexdigest()[:24]
                collision, same_answer, no_op, expected_norm = False, True, True, 0.0
                basis_hash = representation["basis_sha256"]
                layer, position, rank, strength, control_id = 2, 14, 2, 0.0, None
            else:
                base, source = by_id[pair["base_id"]], by_id[pair["source_id"]]
                stratum = pair["stratum"]
                expected = {key: pair[f"expected_{key}"] for key in ("z2", "z3", "n3", "answer")}
                pair_id = pair["pair_id"]
                collision, same_answer, no_op = pair["collision"], pair["same_observable_answer"], pair["no_op"]
                expected_norm = norms[pair_id]
                if control is None:
                    basis_hash = representation["basis_sha256"]
                    layer, position, rank, strength, control_id = 2, 14, 2, 1.0, None
                else:
                    basis_hash = control["basis_sha256"]
                    layer, position, rank, strength, control_id = control["layer"], control["position"], control["rank"], control["strength"], control["control_id"]
            call_id = "devonly-remcall-" + entropy.derive("call-id", ordinal, 16).hex()
            nonce = "devonly-remnonce-" + entropy.derive("nonce", ordinal, 16).hex()
            row = {
                "schema": "exp33ae.dev_only_registry_row.v2", "namespace": DEV_NAMESPACE,
                "stage_id": entropy.stage_id, "claim_bearing": False, "protocol_sha256": protocol_hash(),
                "operative_manifest_sha256": operative_manifest["manifest_sha256"],
                "environment_sha256": environment["manifest_sha256"],
                "allocation_sha256": allocation_manifest["allocation_sha256"],
                "representation_sha256": representation["representation_sha256"],
                "representation_file_sha256": sha256_file(output / "representation.json"),
                "control_manifest_sha256": controls["control_manifest_sha256"],
                "control_manifest_file_sha256": sha256_file(control_path),
                "training_manifest_sha256": training["scientific_manifest_sha256"],
                "training_manifest_file_sha256": sha256_file(training_path),
                "checkpoint_relative_path": training["checkpoint_relative_path"],
                "checkpoint_file_sha256": training["checkpoint_file_sha256"],
                "checkpoint_tensor_sha256": training["final_tensor_state_sha256"],
                "model_order": model_order, "model_seed_identity": training["model_seed_identity"],
                "order": ordinal, "method": method, "control_id": control_id,
                "programme_id": base.programme_id, "causal_signature": base.causal_signature, "axis": base.axis,
                "base_id": base.example_id, "source_id": source.example_id, "pair_id": pair_id,
                "pair_stratum": stratum, "eligible": True, "collision": collision,
                "same_observable_answer": same_answer, "no_op": no_op,
                "base_z2": base.z2, "base_z3": base.z3, "base_n3": base.n3, "base_answer": base.answer,
                "source_z2": source.z2, "source_z3": source.z3, "source_n3": source.n3, "source_answer": source.answer,
                "expected_z2": expected["z2"], "expected_z3": expected["z3"], "expected_n3": expected["n3"], "expected_answer": expected["answer"],
                "basis_sha256": basis_hash, "decoder_sha256": representation["representation_sha256"],
                "layer": layer, "position": position, "rank": rank, "strength": strength,
                "expected_applied_norm": expected_norm, "norm_tolerance_absolute": 1e-6, "norm_tolerance_relative": 1e-5,
                "call_id": call_id, "nonce": nonce, "planned_attempts": 1,
                "retry_policy": "same-access-token-resume-first-missing-no-reexecution",
            }
            row["registry_row_sha256"] = _row_hash(row)
            registry.append(row)
            ordinal += 1

        for example in clean:
            append(example, None, "clean_task_decode", None)
        for pair in pairs:
            append(by_id[pair["base_id"]], pair, "declared_iit_alignment", None)
        for control in controls["controls"]:
            for pair in A:
                append(by_id[pair["base_id"]], pair, control["family"], control)
    if len(registry) != 11760:
        raise RuntimeError(f"registry count {len(registry)}")
    payload_commitment = object_sha256(registry)
    for row in registry:
        row["registry_payload_commitment"] = payload_commitment
    registry_path = output / "registry" / "registry.jsonl"
    write_jsonl(registry_path, registry)
    manifest = {
        "schema": "exp33ae.dev_only_registry_manifest.v2", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "stage_id": entropy.stage_id, "protocol_sha256": protocol_hash(),
        "model_seed_identities": list(entropy.model_seeds()), "row_count": len(registry),
        "rows_per_model": 3920, "payload_commitment": payload_commitment,
        "registry_file_sha256": sha256_file(registry_path),
        "method_counts": dict(sorted(Counter(row["method"] for row in registry).items())),
        "confirmation_registry_materialized": False,
    }
    manifest["registry_manifest_sha256"] = object_sha256(manifest)
    write_json(output / "registry" / "manifest.json", manifest)
    return {"rows": registry, "manifest": manifest}
