"""Generated protocol schemas; the normative JSON remains the single source."""

from __future__ import annotations

from .normative import DEV_NAMESPACE, protocol_hash


def schemas() -> dict[str, dict]:
    common = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "x-protocol-sha256": protocol_hash(),
        "additionalProperties": True,
    }
    registry_required = [
        "schema", "domain", "protocol_version", "protocol_sha256", "stage_id", "claim_bearing",
        "model_order", "model_seed_identity", "checkpoint_sha256", "checkpoint_tensor_sha256", "readout_sha256",
        "programme_id", "causal_signature", "base_id", "source_id", "pair_id", "pair_stratum", "eligible",
        "collision", "same_observable_answer", "no_op", "base_z2", "base_z3", "base_n3", "base_answer",
        "source_z2", "source_z3", "source_n3", "source_answer", "axis", "method", "control_id",
        "basis_sha256", "layer", "position", "rank", "strength", "applied_norm_reference",
        "norm_tolerance_absolute", "norm_tolerance_relative", "expected_z2", "expected_z3", "expected_n3",
        "expected_answer", "call_id", "nonce", "planned_attempts", "retry_policy", "order",
        "registry_payload_commitment",
    ]
    raw_required = [
        "schema", "namespace", "stage_id", "claim_bearing", "call_id", "nonce", "model_order",
        "model_seed_identity", "checkpoint_sha256", "programme_id", "pair_id", "method", "basis_sha256",
        "layer", "position", "rank", "strength", "attempt_number", "attempted", "completed", "dropped",
        "error", "base_prediction", "source_prediction", "intervened_prediction", "decoded_z2",
        "intervened_z3", "intervened_n3", "applied_norm",
    ]
    return {
        "registry-row.schema.json": {
            **common, "$id": "exp33ae.preconfirmation_registry_row.v1", "type": "object", "required": registry_required,
            "properties": {
                "schema": {"const": "exp33ae.preconfirmation_registry_row.v1"},
                "domain": {"const": DEV_NAMESPACE}, "claim_bearing": {"const": False},
                "eligible": {"type": "boolean"}, "collision": {"type": "boolean"},
                "same_observable_answer": {"type": "boolean"}, "no_op": {"type": "boolean"},
            },
        },
        "raw-call.schema.json": {
            **common, "$id": "exp33ae.dev_only_raw_call.v1", "type": "object", "required": raw_required,
            "properties": {
                "schema": {"const": "exp33ae.dev_only_raw_call.v1"},
                "namespace": {"const": DEV_NAMESPACE}, "claim_bearing": {"const": False},
                "attempted": {"type": "boolean"}, "completed": {"type": "boolean"}, "dropped": {"type": "boolean"},
            },
        },
        "access-event.schema.json": {
            **common, "$id": "exp33ae.dev_only_access_event.v1", "type": "object",
            "required": ["schema", "namespace", "stage_id", "ordinal", "event", "details", "previous_event_hash", "event_hash"],
            "properties": {"namespace": {"const": DEV_NAMESPACE}},
        },
        "simulated-decision.schema.json": {
            **common, "$id": "exp33ae.dev_only_simulated_decision.v1", "type": "object",
            "required": ["schema", "namespace", "claim_bearing", "simulated_status", "faults", "terminal_status_materialized"],
            "properties": {"namespace": {"const": DEV_NAMESPACE}, "claim_bearing": {"const": False}, "terminal_status_materialized": {"const": False}},
        },
    }
