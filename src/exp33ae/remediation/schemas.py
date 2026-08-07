"""Strict claim-shaped schemas mirrored by invoked exhaustive validators."""

from __future__ import annotations

from .normative import DEV_NAMESPACE, protocol_hash


REGISTRY_KEYS = {
    "schema", "namespace", "stage_id", "claim_bearing", "protocol_sha256",
    "operative_manifest_sha256", "environment_sha256", "allocation_sha256",
    "representation_sha256", "representation_file_sha256", "control_manifest_sha256",
    "control_manifest_file_sha256", "training_manifest_sha256", "training_manifest_file_sha256",
    "checkpoint_relative_path", "checkpoint_file_sha256", "checkpoint_tensor_sha256",
    "model_order", "model_seed_identity", "order", "method", "control_id", "programme_id",
    "causal_signature", "axis", "base_id", "source_id", "pair_id", "pair_stratum", "eligible",
    "collision", "same_observable_answer", "no_op", "base_z2", "base_z3", "base_n3", "base_answer",
    "source_z2", "source_z3", "source_n3", "source_answer", "expected_z2", "expected_z3",
    "expected_n3", "expected_answer", "basis_sha256", "decoder_sha256", "layer", "position",
    "rank", "strength", "expected_applied_norm", "norm_tolerance_absolute", "norm_tolerance_relative",
    "call_id", "nonce", "planned_attempts", "retry_policy", "registry_row_sha256",
    "registry_payload_commitment",
}

RAW_KEYS = {
    "schema", "namespace", "stage_id", "claim_bearing", "call_id", "nonce", "order",
    "registry_row_sha256", "registry_payload_commitment", "model_order", "model_seed_identity",
    "checkpoint_file_sha256", "checkpoint_tensor_sha256", "operative_manifest_sha256",
    "environment_sha256", "allocation_sha256", "representation_sha256", "control_manifest_sha256",
    "training_manifest_sha256", "programme_id", "causal_signature", "base_id", "source_id", "pair_id",
    "pair_stratum", "method", "control_id", "basis_sha256", "decoder_sha256", "actual_layer",
    "actual_position", "actual_rank", "actual_strength", "attempt_number", "attempted", "completed",
    "dropped", "error", "base_prediction", "source_prediction", "intervened_prediction", "decoded_z2",
    "intervened_z3", "intervened_n3", "applied_norm", "expected_applied_norm", "norm_absolute_error",
    "base_hidden_norm", "activation_delta_sha256", "hook_consumed_by_downstream", "raw_row_sha256",
}


def _schema(identifier: str, keys: set[str], properties: dict) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": identifier,
        "x-protocol-sha256": protocol_hash(),
        "type": "object",
        "additionalProperties": False,
        "required": sorted(keys),
        "properties": properties,
    }


def generated_schemas() -> dict[str, dict]:
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    registry_properties = {key: {"type": "string", "minLength": 1} for key in REGISTRY_KEYS}
    for key in {
        "model_order", "model_seed_identity", "order", "base_z2", "base_z3", "base_n3",
        "base_answer", "source_z2", "source_z3", "source_n3", "source_answer", "expected_z2",
        "expected_z3", "expected_n3", "expected_answer", "layer", "position", "rank", "planned_attempts",
    }:
        registry_properties[key] = {"type": "integer"}
    for key in {"strength", "expected_applied_norm", "norm_tolerance_absolute", "norm_tolerance_relative"}:
        registry_properties[key] = {"type": "number"}
    for key in {"claim_bearing", "eligible", "collision", "same_observable_answer", "no_op"}:
        registry_properties[key] = {"type": "boolean"}
    registry_properties["control_id"] = {"type": ["string", "null"]}
    registry_properties.update({
        "schema": {"const": "exp33ae.dev_only_registry_row.v2"}, "namespace": {"const": DEV_NAMESPACE},
        "claim_bearing": {"const": False}, "protocol_sha256": {"const": protocol_hash()},
        "model_order": {"type": "integer", "minimum": 0, "maximum": 2},
        "model_seed_identity": {"type": "integer", "minimum": 1}, "order": {"type": "integer", "minimum": 0, "maximum": 11759},
        "eligible": {"type": "boolean"}, "collision": {"type": "boolean"},
        "same_observable_answer": {"type": "boolean"}, "no_op": {"type": "boolean"},
        "layer": {"type": "integer", "minimum": 0, "maximum": 3}, "position": {"type": "integer", "minimum": 0, "maximum": 16},
        "rank": {"const": 2}, "strength": {"type": "number", "minimum": 0, "maximum": 1},
    })
    for key in ("protocol_sha256", "operative_manifest_sha256", "environment_sha256", "allocation_sha256", "representation_sha256", "representation_file_sha256", "control_manifest_sha256", "control_manifest_file_sha256", "training_manifest_sha256", "training_manifest_file_sha256", "checkpoint_file_sha256", "checkpoint_tensor_sha256", "causal_signature", "basis_sha256", "decoder_sha256", "registry_row_sha256", "registry_payload_commitment"):
        registry_properties[key] = sha
    raw_properties = {key: {"type": "string", "minLength": 1} for key in RAW_KEYS}
    for key in {
        "order", "model_order", "model_seed_identity", "actual_layer", "actual_position", "actual_rank",
        "attempt_number", "base_prediction", "source_prediction", "intervened_prediction", "decoded_z2",
        "intervened_z3", "intervened_n3",
    }:
        raw_properties[key] = {"type": "integer"}
    for key in {"actual_strength", "applied_norm", "expected_applied_norm", "norm_absolute_error", "base_hidden_norm"}:
        raw_properties[key] = {"type": "number"}
    for key in {"claim_bearing", "attempted", "completed", "dropped", "hook_consumed_by_downstream"}:
        raw_properties[key] = {"type": "boolean"}
    raw_properties["control_id"] = {"type": ["string", "null"]}
    raw_properties["error"] = {"type": ["string", "null"]}
    raw_properties.update({
        "schema": {"const": "exp33ae.dev_only_raw_inference.v2"}, "namespace": {"const": DEV_NAMESPACE},
        "claim_bearing": {"const": False}, "attempted": {"const": True}, "completed": {"const": True},
        "dropped": {"const": False}, "attempt_number": {"const": 1},
        "hook_consumed_by_downstream": {"type": "boolean"},
    })
    for key in ("registry_row_sha256", "registry_payload_commitment", "checkpoint_file_sha256", "checkpoint_tensor_sha256", "operative_manifest_sha256", "environment_sha256", "allocation_sha256", "representation_sha256", "control_manifest_sha256", "training_manifest_sha256", "causal_signature", "basis_sha256", "decoder_sha256", "activation_delta_sha256", "raw_row_sha256"):
        raw_properties[key] = sha
    return {
        "registry-row.schema.json": _schema("exp33ae.dev_only_registry_row.v2", REGISTRY_KEYS, registry_properties),
        "raw-inference.schema.json": _schema("exp33ae.dev_only_raw_inference.v2", RAW_KEYS, raw_properties),
    }
