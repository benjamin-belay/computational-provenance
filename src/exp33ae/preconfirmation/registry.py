"""Build and reconcile the exact 11,760-row DEV_ONLY registry fixture."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256

from ..artifacts import canonical_json
from .entropy import DevEntropy
from .normative import DEV_NAMESPACE, load_protocol, protocol_hash
from .representation import primary_basis
from ..artifacts import tensor_sha256


def _identity_hash(label: str, stage: str, seed: int) -> str:
    return sha256(canonical_json(["DEV_ONLY_FIXTURE_IDENTITY", label, stage, seed]).encode()).hexdigest()


def _base_row(entropy: DevEntropy, ordinal: int, model_order: int, seed: int, example, pair: dict | None, method: str, control: dict | None) -> dict:
    clean = pair is None
    base_id = example.example_id if clean else pair["base_id"]
    source_id = base_id if clean else pair["source_id"]
    call_id = "devonly-pcrcall-" + entropy.derive("call-id", ordinal, 16).hex()
    nonce = "devonly-pcrnonce-" + entropy.derive("nonce", ordinal, 16).hex()
    primary_hash = tensor_sha256(primary_basis())
    if clean:
        basis_hash, layer, position, rank, strength = _identity_hash("clean-no-intervention", entropy.stage_id, seed), None, None, 0, 0.0
        stratum, eligible, no_op = "clean_task_decode", True, True
        expected = {"z2": example.z2, "z3": example.z3, "n3": example.n3, "answer": example.answer}
        base_values = source_values = {"z2": example.z2, "z3": example.z3, "n3": example.n3, "answer": example.answer}
        pair_id = "devonly-clean-pair-" + sha256(canonical_json([seed, example.example_id]).encode()).hexdigest()[:24]
        axis = example.axis
    else:
        basis_hash = primary_hash if control is None else control["basis_sha256"]
        layer = 2 if control is None else control["layer"]
        position = 14 if control is None else control["position"]
        rank, strength = 2, 1.0
        stratum, eligible, no_op = pair["stratum"], pair["eligible"], pair["no_op"]
        expected = {key: pair[f"expected_{key}"] for key in ("z2", "z3", "n3", "answer")}
        base_values = {key: pair[f"base_{key}"] for key in ("z2", "z3", "n3", "answer")}
        source_values = {key: pair[f"source_{key}"] for key in ("z2", "z3", "n3", "answer")}
        pair_id, axis = pair["pair_id"], pair["base_axis"]
    return {
        "schema": "exp33ae.preconfirmation_registry_row.v1", "domain": DEV_NAMESPACE,
        "protocol_version": 1, "protocol_sha256": protocol_hash(), "stage_id": entropy.stage_id,
        "claim_bearing": False, "model_order": model_order, "model_seed_identity": seed,
        "checkpoint_sha256": _identity_hash("checkpoint-fixture", entropy.stage_id, seed),
        "checkpoint_tensor_sha256": _identity_hash("checkpoint-tensor-fixture", entropy.stage_id, seed),
        "readout_sha256": _identity_hash("subspace-readout-fixture", entropy.stage_id, seed),
        "programme_id": example.programme_id if clean else pair["base_programme_id"],
        "causal_signature": example.causal_signature,
        "base_id": base_id, "source_id": source_id, "pair_id": pair_id, "pair_stratum": stratum,
        "eligible": eligible, "collision": False if clean else pair["collision"],
        "same_observable_answer": True if clean else pair["same_observable_answer"], "no_op": no_op,
        "base_z2": base_values["z2"], "base_z3": base_values["z3"], "base_n3": base_values["n3"], "base_answer": base_values["answer"],
        "source_z2": source_values["z2"], "source_z3": source_values["z3"], "source_n3": source_values["n3"], "source_answer": source_values["answer"],
        "axis": axis, "method": method, "control_id": None if control is None else control["control_id"],
        "basis_sha256": basis_hash, "layer": layer, "position": position, "rank": rank, "strength": strength,
        "applied_norm_reference": None if clean else "paired_primary_A_call",
        "norm_tolerance_absolute": 1e-6, "norm_tolerance_relative": 1e-5,
        "expected_z2": expected["z2"], "expected_z3": expected["z3"], "expected_n3": expected["n3"], "expected_answer": expected["answer"],
        "call_id": call_id, "nonce": nonce, "planned_attempts": 1,
        "retry_policy": "same_consumed_session_first_missing_only_no_output_read", "order": ordinal,
    }


def build_registry(allocation: dict, pairs: list[dict], controls: list[dict], entropy: DevEntropy) -> dict:
    protocol = load_protocol()
    clean = [row for row in allocation["examples"] if row.partition in {"clean_target_axis", "clean_parallel_axis"}]
    by_id = {row.example_id: row for row in allocation["examples"]}
    answer_pairs = [row for row in pairs if row["stratum"] == "A_answer_changing"]
    rows, ordinal = [], 0
    seeds = entropy.ordered_model_seeds()
    for model_order, seed in enumerate(seeds):
        for example in clean:
            rows.append(_base_row(entropy, ordinal, model_order, seed, example, None, "clean_task_decode", None)); ordinal += 1
        for pair in pairs:
            rows.append(_base_row(entropy, ordinal, model_order, seed, by_id[pair["base_id"]], pair, "declared_iit_alignment", None)); ordinal += 1
        for control in controls:
            for pair in answer_pairs:
                rows.append(_base_row(entropy, ordinal, model_order, seed, by_id[pair["base_id"]], pair, control["family"], control)); ordinal += 1
    if len(rows) != protocol["registry"]["total"]:
        raise RuntimeError(f"registry row count {len(rows)} != 11760")
    payload_commitment = sha256(canonical_json(rows).encode()).hexdigest()
    for row in rows:
        row["registry_payload_commitment"] = payload_commitment
    ids = [row["call_id"] for row in rows]
    nonces = [row["nonce"] for row in rows]
    if len(set(ids)) != len(rows) or len(set(nonces)) != len(rows):
        raise RuntimeError("call ID or nonce collision")
    per_model = Counter(row["model_seed_identity"] for row in rows)
    if set(per_model.values()) != {3920} or list(per_model) != list(seeds):
        raise RuntimeError("model registry count/order mismatch")
    return {
        "schema": "exp33ae.preconfirmation_registry_fixture.v1", "namespace": DEV_NAMESPACE,
        "stage_id": entropy.stage_id, "claim_bearing": False, "model_seed_identities": list(seeds),
        "payload_commitment": payload_commitment, "rows": rows,
        "file_commitment_rule": "SHA256 canonical JSONL after payload commitment is attached; file hash stored externally",
        "confirmation_registry_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }


def reconcile(plans: list[dict], raw: list[dict]) -> dict:
    plan_ids = [row.get("call_id") for row in plans]
    raw_ids = [row.get("call_id") for row in raw]
    missing = set(plan_ids) - set(raw_ids)
    unplanned = set(raw_ids) - set(plan_ids)
    duplicates = len(raw_ids) - len(set(raw_ids))
    failed = sum(row.get("completed") is False for row in raw)
    retried = sum(row.get("attempt_number") != 1 for row in raw)
    dropped = sum(row.get("dropped") is True for row in raw)
    return {
        "planned": len(plans), "attempted": len(raw), "completed": sum(row.get("completed") is True for row in raw),
        "missing": len(missing), "unplanned": len(unplanned), "duplicates": duplicates,
        "failed": failed, "retried": retried, "dropped": dropped,
        "exact": len(plans) == len(raw) == 11760 and not missing and not unplanned and not duplicates and not failed and not retried and not dropped,
    }
