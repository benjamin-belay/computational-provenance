"""Invoked exhaustive validators for every registry and raw row."""

from __future__ import annotations

from collections import Counter
import math

from .normative import DEV_NAMESPACE, protocol_hash
from .schemas import RAW_KEYS, REGISTRY_KEYS
from .strict import expect_exact_keys, expect_int, expect_number, expect_sha256, expect_type, object_sha256, StrictValidationError


METHODS = {"clean_task_decode", "declared_iit_alignment", "random_direction", "orthogonal_direction", "alternate_location", "alternate_early_layer"}
STRATA = {"clean_task_decode", "A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer"}
AXES = {"blinded_within_run_target_composition", "blinded_within_run_parallel_composition"}


def _nonempty_string(value, label: str, prefix: str | None = None) -> str:
    expect_type(value, str, label)
    if not value or (prefix is not None and not value.startswith(prefix)):
        raise StrictValidationError("SCHEMA", f"{label} malformed")
    return value


def validate_registry(rows: list[dict], manifest: dict, expected_seeds: list[int]) -> None:
    if type(rows) is not list or len(rows) != 11760:
        raise StrictValidationError("REGISTRY_STRUCTURE", "exact registry count")
    if type(expected_seeds) is not list or len(expected_seeds) != 3 or len(set(expected_seeds)) != 3 or any(type(value) is not int or value <= 0 for value in expected_seeds):
        raise StrictValidationError("SEED_IDENTITY", "ordered unique seeds")
    payload_rows = []
    calls, nonces = [], []
    expected_manifest_keys = {
        "schema", "namespace", "claim_bearing", "stage_id", "protocol_sha256",
        "model_seed_identities", "row_count", "rows_per_model", "payload_commitment",
        "registry_file_sha256", "method_counts", "confirmation_registry_materialized",
        "registry_manifest_sha256",
    }
    expect_exact_keys(manifest, expected_manifest_keys, "registry manifest")
    if (
        manifest["schema"] != "exp33ae.dev_only_registry_manifest.v2"
        or manifest["namespace"] != DEV_NAMESPACE
        or manifest["claim_bearing"] is not False
        or manifest["confirmation_registry_materialized"] is not False
        or manifest["protocol_sha256"] != protocol_hash()
        or manifest["model_seed_identities"] != expected_seeds
        or manifest["row_count"] != 11760
        or manifest["rows_per_model"] != 3920
    ):
        raise StrictValidationError("REGISTRY_STRUCTURE", "manifest identity/counts")
    expect_sha256(manifest["payload_commitment"], "registry manifest payload")
    expect_sha256(manifest["registry_file_sha256"], "registry manifest file")
    expect_sha256(manifest["registry_manifest_sha256"], "registry manifest self")
    _nonempty_string(manifest["stage_id"], "registry manifest stage", "DEV_ONLY_")
    if type(manifest["method_counts"]) is not dict or any(type(key) is not str or type(value) is not int for key, value in manifest["method_counts"].items()):
        raise StrictValidationError("SCHEMA", "registry method counts")
    for ordinal, row in enumerate(rows):
        expect_exact_keys(row, REGISTRY_KEYS, f"registry[{ordinal}]")
        if row["schema"] != "exp33ae.dev_only_registry_row.v2" or row["namespace"] != DEV_NAMESPACE or row["claim_bearing"] is not False or row["protocol_sha256"] != protocol_hash():
            raise StrictValidationError("SCHEMA", f"registry[{ordinal}] identity")
        _nonempty_string(row["stage_id"], f"registry[{ordinal}].stage_id", "DEV_ONLY_")
        for field in ("checkpoint_relative_path", "programme_id", "axis", "base_id", "source_id", "pair_id", "pair_stratum", "method", "call_id", "nonce", "retry_policy"):
            _nonempty_string(row[field], f"registry[{ordinal}].{field}")
        if not row["call_id"].startswith("devonly-remcall-") or not row["nonce"].startswith("devonly-remnonce-"):
            raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] call/nonce namespace")
        if row["control_id"] is not None:
            _nonempty_string(row["control_id"], f"registry[{ordinal}].control_id")
        if row["order"] != ordinal or row["model_order"] != ordinal // 3920 or row["model_seed_identity"] != expected_seeds[row["model_order"]]:
            raise StrictValidationError("SEED_IDENTITY", f"registry[{ordinal}] order/seed")
        if row["method"] not in METHODS or row["pair_stratum"] not in STRATA or row["axis"] not in AXES:
            raise StrictValidationError("SCHEMA", f"registry[{ordinal}] enum")
        for field in ("eligible", "collision", "same_observable_answer", "no_op"):
            if type(row[field]) is not bool:
                raise StrictValidationError("SCHEMA", f"registry[{ordinal}].{field}")
        for field in ("model_order", "model_seed_identity", "order", "base_z2", "base_z3", "base_n3", "base_answer", "source_z2", "source_z3", "source_n3", "source_answer", "expected_z2", "expected_z3", "expected_n3", "expected_answer", "layer", "position", "rank", "planned_attempts"):
            expect_int(row[field], f"registry[{ordinal}].{field}")
        for field in ("base_z2", "base_z3", "base_n3", "source_z2", "source_z3", "source_n3", "expected_z2", "expected_z3", "expected_n3"):
            if row[field] not in {0, 1}:
                raise StrictValidationError("SCHEMA", f"registry[{ordinal}].{field} range")
        for field in ("base_answer", "source_answer", "expected_answer"):
            if row[field] not in {0, 1, 2, 3}:
                raise StrictValidationError("SCHEMA", f"registry[{ordinal}].{field} range")
        if row["rank"] != 2 or row["planned_attempts"] != 1 or row["layer"] not in range(4) or row["position"] not in range(17):
            raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] site/rank/attempt")
        if (
            row["stage_id"] != manifest["stage_id"]
            or row["eligible"] is not True
            or row["retry_policy"] != "same-access-token-resume-first-missing-no-reexecution"
            or row["checkpoint_relative_path"] != f"checkpoints/model-{row['model_order']}.pt"
            or row["norm_tolerance_absolute"] != 1e-6
            or row["norm_tolerance_relative"] != 1e-5
        ):
            raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] frozen policy")
        if row["method"] == "clean_task_decode":
            if (row["control_id"], row["layer"], row["position"], row["strength"], row["expected_applied_norm"]) != (None, 2, 14, 0.0, 0.0):
                raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] clean plan")
        elif row["method"] == "declared_iit_alignment":
            if (row["control_id"], row["layer"], row["position"], row["strength"]) != (None, 2, 14, 1.0):
                raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] primary plan")
        elif row["control_id"] is None or row["strength"] != 1.0:
            raise StrictValidationError("REGISTRY_STRUCTURE", f"registry[{ordinal}] control plan")
        for field in ("strength", "expected_applied_norm", "norm_tolerance_absolute", "norm_tolerance_relative"):
            expect_number(row[field], f"registry[{ordinal}].{field}", 0.0)
        for field in ("protocol_sha256", "operative_manifest_sha256", "environment_sha256", "allocation_sha256", "representation_sha256", "representation_file_sha256", "control_manifest_sha256", "control_manifest_file_sha256", "training_manifest_sha256", "training_manifest_file_sha256", "checkpoint_file_sha256", "checkpoint_tensor_sha256", "causal_signature", "basis_sha256", "decoder_sha256", "registry_row_sha256", "registry_payload_commitment"):
            expect_sha256(row[field], f"registry[{ordinal}].{field}")
        copy = dict(row); observed_hash = copy.pop("registry_row_sha256"); copy.pop("registry_payload_commitment")
        if object_sha256(copy) != observed_hash:
            raise StrictValidationError("REGISTRY_BINDING", f"registry[{ordinal}] row hash")
        payload = dict(row); payload.pop("registry_payload_commitment")
        payload_rows.append(payload)
        calls.append(row["call_id"]); nonces.append(row["nonce"])
    commitment = object_sha256(payload_rows)
    if commitment != manifest["payload_commitment"] or any(row["registry_payload_commitment"] != commitment for row in rows):
        raise StrictValidationError("REGISTRY_BINDING", "payload commitment")
    if len(set(calls)) != len(calls) or len(set(nonces)) != len(nonces):
        raise StrictValidationError("REGISTRY_STRUCTURE", "duplicate call or nonce")
    for order, seed in enumerate(expected_seeds):
        model = rows[order*3920:(order+1)*3920]
        methods = Counter(row["method"] for row in model)
        if methods != Counter({"clean_task_decode": 1600, "declared_iit_alignment": 400, "random_direction": 640, "orthogonal_direction": 640, "alternate_location": 480, "alternate_early_layer": 160}):
            raise StrictValidationError("REGISTRY_STRUCTURE", f"model {order} method counts")
        strata = Counter(row["pair_stratum"] for row in model if row["method"] == "declared_iit_alignment")
        if strata != Counter({"A_answer_changing": 160, "B_outcome_collision": 80, "C_same_state": 80, "D_same_observable_answer": 80}):
            raise StrictValidationError("REGISTRY_STRUCTURE", f"model {order} strata")
        for family, count in (("random_direction", 4), ("orthogonal_direction", 4), ("alternate_location", 3), ("alternate_early_layer", 1)):
            ids = Counter(row["control_id"] for row in model if row["method"] == family)
            if len(ids) != count or set(ids.values()) != {160}:
                raise StrictValidationError("REGISTRY_STRUCTURE", f"model {order} controls {family}")
    expected_methods = Counter(row["method"] for row in rows)
    if manifest["method_counts"] != dict(sorted(expected_methods.items())):
        raise StrictValidationError("REGISTRY_STRUCTURE", "manifest method counts")


def validate_raw(rows: list[dict], registry: list[dict]) -> None:
    if type(rows) is not list or len(rows) != 11760:
        raise StrictValidationError("CALL_RECONCILIATION", "raw exact count")
    for ordinal, (row, plan) in enumerate(zip(rows, registry)):
        expect_exact_keys(row, RAW_KEYS, f"raw[{ordinal}]")
        if row["schema"] != "exp33ae.dev_only_raw_inference.v2" or row["namespace"] != DEV_NAMESPACE or row["claim_bearing"] is not False:
            raise StrictValidationError("SCHEMA", f"raw[{ordinal}] identity")
        _nonempty_string(row["stage_id"], f"raw[{ordinal}].stage_id", "DEV_ONLY_")
        for field in ("call_id", "nonce", "programme_id", "base_id", "source_id", "pair_id", "pair_stratum", "method"):
            _nonempty_string(row[field], f"raw[{ordinal}].{field}")
        if row["control_id"] is not None:
            _nonempty_string(row["control_id"], f"raw[{ordinal}].control_id")
        if row["error"] is not None:
            raise StrictValidationError("CALL_RECONCILIATION", f"raw[{ordinal}] error")
        if type(row["hook_consumed_by_downstream"]) is not bool:
            raise StrictValidationError("SCHEMA", f"raw[{ordinal}].hook_consumed_by_downstream")
        if row["method"] == "clean_task_decode":
            if row["hook_consumed_by_downstream"] is not False:
                raise StrictValidationError("RAW_STRUCTURE", f"raw[{ordinal}] clean hook")
        elif row["hook_consumed_by_downstream"] is not True:
            raise StrictValidationError("RAW_STRUCTURE", f"raw[{ordinal}] intervention hook")
        if row["attempted"] is not True or row["completed"] is not True or row["dropped"] is not False or row["attempt_number"] != 1 or row["error"] is not None:
            raise StrictValidationError("CALL_RECONCILIATION", f"raw[{ordinal}] completion")
        identity = {
            "stage_id": "stage_id", "call_id": "call_id", "nonce": "nonce", "order": "order", "registry_row_sha256": "registry_row_sha256",
            "registry_payload_commitment": "registry_payload_commitment", "model_order": "model_order", "model_seed_identity": "model_seed_identity",
            "checkpoint_file_sha256": "checkpoint_file_sha256", "checkpoint_tensor_sha256": "checkpoint_tensor_sha256",
            "operative_manifest_sha256": "operative_manifest_sha256", "environment_sha256": "environment_sha256",
            "allocation_sha256": "allocation_sha256", "representation_sha256": "representation_sha256",
            "control_manifest_sha256": "control_manifest_sha256", "training_manifest_sha256": "training_manifest_sha256",
            "programme_id": "programme_id", "causal_signature": "causal_signature", "base_id": "base_id", "source_id": "source_id",
            "pair_id": "pair_id", "pair_stratum": "pair_stratum", "method": "method", "control_id": "control_id",
            "basis_sha256": "basis_sha256", "decoder_sha256": "decoder_sha256", "actual_layer": "layer",
            "actual_position": "position", "actual_rank": "rank", "actual_strength": "strength",
            "expected_applied_norm": "expected_applied_norm",
        }
        if any(row[raw_name] != plan[plan_name] for raw_name, plan_name in identity.items()):
            raise StrictValidationError("RAW_BINDING", f"raw[{ordinal}] identity")
        for field in ("base_prediction", "source_prediction", "intervened_prediction"):
            expect_int(row[field], f"raw[{ordinal}].{field}", 0, 3)
        for field in ("decoded_z2", "intervened_z3", "intervened_n3"):
            expect_int(row[field], f"raw[{ordinal}].{field}", 0, 1)
        for field in ("applied_norm", "expected_applied_norm", "norm_absolute_error", "base_hidden_norm"):
            expect_number(row[field], f"raw[{ordinal}].{field}", 0.0)
        for field in ("checkpoint_file_sha256", "checkpoint_tensor_sha256", "operative_manifest_sha256", "environment_sha256", "allocation_sha256", "representation_sha256", "control_manifest_sha256", "training_manifest_sha256", "causal_signature", "basis_sha256", "decoder_sha256", "activation_delta_sha256", "raw_row_sha256"):
            expect_sha256(row[field], f"raw[{ordinal}].{field}")
        copy = dict(row); observed = copy.pop("raw_row_sha256")
        if object_sha256(copy) != observed:
            raise StrictValidationError("RAW_BINDING", f"raw[{ordinal}] row hash")
        tolerance = plan["norm_tolerance_absolute"] + plan["norm_tolerance_relative"] * abs(plan["expected_applied_norm"])
        if abs(row["applied_norm"] - plan["expected_applied_norm"]) > tolerance:
            raise StrictValidationError("NORM_MISMATCH", f"raw[{ordinal}] applied norm")
