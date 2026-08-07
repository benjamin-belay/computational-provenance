#!/usr/bin/env python3
"""Static qualification for the position-indexed RNG2 protocol."""
from __future__ import annotations

import argparse
import ast
import hashlib
import hmac
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol/csw-am-scb-m1-rng2-s0"
CONFIG_PATH = PROTOCOL / "CONFIG.json"
SAMPLER_PATH = PROTOCOL / "SAMPLER_PROTOCOL.json"
NUMERIC_PATH = PROTOCOL / "NUMERIC_AND_SCHEDULE_PROTOCOL.json"
VALIDATION_PATH = PROTOCOL / "VALIDATION_PROTOCOL.json"
ALLOCATION_PATH = PROTOCOL / "ALLOCATION_PROTOCOL.json"
SOURCE_FILES = (
    "protocol/csw-am-scb-m1-rng2-s0/CONFIG.json",
    "protocol/csw-am-scb-m1-rng2-s0/SAMPLER_PROTOCOL.json",
    "protocol/csw-am-scb-m1-rng2-s0/NUMERIC_AND_SCHEDULE_PROTOCOL.json",
    "protocol/csw-am-scb-m1-rng2-s0/VALIDATION_PROTOCOL.json",
    "protocol/csw-am-scb-m1-rng2-s0/ALLOCATION_PROTOCOL.json",
    "tools/csw_am_scb_m1_rng2_s0.py",
    "tools/csw_am_scb_m1_rng2_s0_reference.py",
    "tests/test_csw_am_scb_m1_rng2_s0.py",
)
DOMAIN = b"EXP33A|SCB-M1|RNG2|POSITION-UNIFORM|V1"


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("DUPLICATE_KEY:" + key)
        result[key] = value
    return result


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def is_hex64(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_index(index: dict[str, Any]) -> None:
    if set(index) != {
        "protocol_version", "sampler_domain", "pairing_context_id",
        "closure_id", "output_index", "model_step", "subdraw_index",
    }:
        raise ValueError("INDEX_FIELDS")
    if (
        index["protocol_version"] != "RNG2_V1"
        or index["sampler_domain"] != "GENERIC_STOCHASTIC_CHANNEL"
        or not is_hex64(index["pairing_context_id"])
        or not is_hex64(index["closure_id"])
        or type(index["output_index"]) is not int
        or index["output_index"] not in range(8)
        or type(index["model_step"]) is not int
        or index["model_step"] not in range(49)
        or index["subdraw_index"] != 0
    ):
        raise ValueError("INDEX_VALUE")


def position_uniform(key: bytes, index: dict[str, Any]) -> tuple[float, str]:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("KEY")
    validate_index(index)
    digest = hmac.new(key, DOMAIN + b"\x00" + canonical(index), hashlib.sha256).digest()
    integer52 = int.from_bytes(digest[:8], "big") >> 12
    uniform = (integer52 + 0.5) / (1 << 52)
    if not 0.0 < uniform < 1.0:
        raise ValueError("UNIFORM_RANGE")
    return uniform, digest.hex()


def inverse_cdf(probabilities: Sequence[float], uniform: float) -> tuple[int, bool]:
    if (
        not probabilities
        or type(uniform) is not float
        or not math.isfinite(uniform)
        or not 0.0 <= uniform <= 1.0
    ):
        raise ValueError("CDF_INPUT")
    values = [float(value) for value in probabilities]
    if (
        any(not math.isfinite(value) or value < 0.0 for value in values)
        or math.fsum(values) <= 0.0
    ):
        raise ValueError("PROBABILITY")
    total = math.fsum(values)
    normalized = [value / total for value in values]
    cumulative = 0.0
    for index, probability in enumerate(normalized):
        cumulative += probability
        if uniform <= cumulative:
            return index, False
    return len(values) - 1, True


def load_protocol() -> tuple[dict[str, Any], ...]:
    return tuple(read_json(path) for path in (
        CONFIG_PATH, SAMPLER_PATH, NUMERIC_PATH, VALIDATION_PATH, ALLOCATION_PATH
    ))


def source_schedule_census() -> dict[str, Any]:
    source = (
        ROOT / "tools/csw_a0_stochastic_channel.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    generate = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    sequence_lengths = []
    for node in ast.walk(generate):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "sequence"
            and isinstance(node.value, ast.List)
        ):
            sequence_lengths.append(len(node.value.elts))
    sample_calls = [
        node for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sample_ids"
    ]
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_s0.source_schedule.v1",
        "frozen_source_sha256": file_sha(ROOT / "tools/csw_a0_stochastic_channel.py"),
        "order_sampling_call_sites": 1,
        "field_sampling_call_sites": 1,
        "sample_ids_call_sites_in_generate": len(sample_calls),
        "field_sequence_lengths": sequence_lengths,
        "registered_fields": 4,
        "derived_runtime_draws": 1 + 4 * sequence_lengths[0]
        if sequence_lengths == [12] else None,
        "none_appended_to_model_history": 'model_history.append(selected)' in source,
        "none_excluded_from_public_tokens": (
            'if selected != "<NONE>":' in source
            and "public_tokens.append(selected)" in source
        ),
        "history_conditioned_predecessor_uniform": (
            "def counter_uniform(stream_seed: bytes, step: int, model_history:"
            in source
        ),
    }


def validate_lineage(config: dict[str, Any]) -> None:
    for name in (
        "closed_agg_d0_audit", "closed_agg_d0_terminal", "rng2_boundary",
        "resource_status", "frozen_a0_realiser",
    ):
        path = ROOT / config["lineage"][f"{name}_path"]
        if file_sha(path) != config["lineage"][f"{name}_sha256"]:
            raise ValueError("LINEAGE_HASH:" + name)
    audit = read_json(ROOT / config["lineage"]["closed_agg_d0_audit_path"])
    terminal = read_json(ROOT / config["lineage"]["closed_agg_d0_terminal_path"])
    boundary = read_json(ROOT / config["lineage"]["rng2_boundary_path"])
    if (
        audit["main_verdict"] != config["lineage"]["closed_agg_d0_terminal"]
        or terminal["terminal"] != config["lineage"]["closed_agg_d0_terminal"]
        or boundary["status"] != "DRAFT_NON_AUTHORIZING"
        or boundary["successor_authorized"] is not False
        or boundary["new_stochastic_execution_authorized"] is not False
    ):
        raise ValueError("CLOSED_BOUNDARY")


def validate_protocol() -> dict[str, Any]:
    config, sampler, numeric, validation, allocation = load_protocol()
    validate_lineage(config)
    static = config["static_accounting"]
    if any(static.values()):
        raise ValueError("STATIC_ACCOUNTING")
    prf = sampler["prf"]
    schedule = numeric["draw_schedule"]
    none = numeric["none_and_termination"]
    gates = {
        "new_channel": sampler["identity"]
        == "RNG2_POSITION_INDEXED_HMAC_SHA256_MIDPOINT52_V1",
        "autoregression_preserved": (
            sampler["autoregression"]["probability_distribution_depends_on_current_generated_history"]
            and sampler["autoregression"]["only_the_primitive_uniform_is_history_independent"]
        ),
        "index_excludes_state_role_history_and_receipts": all(
            field in sampler["canonical_index"]["prohibited_fields"]
            for field in (
                "natural_or_alias_role", "z2", "generated_history",
                "receipt_id", "exact_execution_id", "exact_run_id",
            )
        ),
        "shared_pairing_context": (
            sampler["pairing"]["same_pairing_context_closure_output_step_subdraw_implies_same_uniform"]
            and sampler["pairing"]["pairing_context_is_distinct_from_exact_run_and_execution_identity"]
        ),
        "open_binary64_uniform": (
            prf["digest_bits_used"] == "FIRST_52_MOST_SIGNIFICANT_BITS"
            and prf["uniform_range"] == "OPEN_INTERVAL_0_1"
        ),
        "fixed_49_step_schedule": (
            schedule["total_model_steps"] == 49
            and schedule["order_steps"] + schedule["fields"]
            * schedule["operations_per_field"] == 49
            and schedule["subdraws_per_step"] == 1
            and schedule["subdraw_index"] == 0
        ),
        "no_branch_dependent_randomness": all(
            schedule[key] == "PROHIBITED"
            for key in (
                "branch_dependent_draw_consumption", "rejection_sampling",
                "result_dependent_resampling", "parser_repair_randomness",
                "tie_break_randomness",
            )
        ),
        "none_semantics_complete": (
            none["NONE_is_appended_to_model_history"]
            and none["NONE_is_not_appended_to_public_tokens"]
            and none["model_advances_on_NONE"]
            and none["NONE_does_not_terminate_generation"]
            and none["all_49_uniforms_are_defined_and_consumed"]
            and none["EOS_is_not_a_candidate_in_the_49_step_schedule"]
        ),
        "distributional_validation_frozen": (
            validation["distributional_test"]["contexts_per_fixture"] == 262144
            and len(validation["distributional_fixtures"]) == 8
            and not validation["generic_channel_requalification"]["A0_Q1_power_inherited"]
        ),
        "formal_migration_required": (
            allocation["status"] == "FORMAL_MIGRATION_REQUIRED_BEFORE_ANY_RNG2_GENERATION"
            and allocation["known_public_population"]["fresh_unallocated_closures"] == 0
            and "NO_SILENT_ROLE_MIGRATION" in allocation["prohibitions"]
        ),
        "only_r0_authorized": (
            config["authorizations"]["RNG2_R0_after_ready"]
            and not any(
                value for key, value in config["authorizations"].items()
                if key != "RNG2_R0_after_ready"
            )
        ),
    }
    return {"config": config, "gates": gates}


def static_vectors() -> dict[str, Any]:
    key = bytes(range(32))
    base = {
        "protocol_version": "RNG2_V1",
        "sampler_domain": "GENERIC_STOCHASTIC_CHANNEL",
        "pairing_context_id": "11" * 32,
        "closure_id": "22" * 32,
        "output_index": 3,
        "model_step": 4,
        "subdraw_index": 0,
    }
    natural, digest = position_uniform(key, base)
    alias, alias_digest = position_uniform(key, dict(base))
    other = dict(base)
    other["pairing_context_id"] = "33" * 32
    separated, separated_digest = position_uniform(key, other)
    cdf = {
        "u_zero": inverse_cdf([0.5, 0.5], 0.0),
        "u_boundary": inverse_cdf([0.5, 0.5], 0.5),
        "u_above_boundary": inverse_cdf([0.5, 0.5], math.nextafter(0.5, 1.0)),
        "u_below_one": inverse_cdf([0.1, 0.2, 0.7], math.nextafter(1.0, 0.0)),
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_s0.static_vectors.v1",
        "key_hex": key.hex(),
        "canonical_index": base,
        "canonical_index_sha256": sha(canonical(base)),
        "uniform": natural,
        "prf_digest": digest,
        "paired_uniform": alias,
        "paired_prf_digest": alias_digest,
        "different_context_uniform": separated,
        "different_context_prf_digest": separated_digest,
        "paired_equal": natural == alias and digest == alias_digest,
        "different_context_separated": digest != separated_digest,
        "cdf_fixtures": {
            key: {"selected_index": value[0], "final_bin_fallback": value[1]}
            for key, value in cdf.items()
        },
    }


def capacity_census() -> dict[str, Any]:
    _, _, _, _, allocation = load_protocol()
    manifest = read_json(
        ROOT / "artifacts/dev_only/csw-am-scb-m1-car-agg-r0/ALLOCATION_MANIFEST.json"
    )
    commitments = read_json(
        ROOT / "artifacts/dev_only/csw-am-scb-m1-car-agg-r0/ROLE_COMMITMENTS.json"
    )
    resources = read_json(
        ROOT / "artifacts/dev_only/csw-am-scb-m1-car-agg-d0-closure/RESOURCE_STATUS.json"
    )
    expected = allocation["known_public_population"]
    role_counts = manifest["role_counts"]
    gates = {
        "population_total": manifest["eligible_closures"]
        == expected["total_eligible_closures"] == 1920,
        "development_released": (
            role_counts["AGG_DEVELOPMENT"] == expected["released_AGG_DEVELOPMENT_closures"]
            and commitments["roles"]["AGG_DEVELOPMENT"]["released_in_r0"]
        ),
        "calibration_unseen": (
            role_counts["AGG_CALIBRATION"] == expected["sealed_AGG_CALIBRATION_closures"]
            == resources["untouched"]["calibration_closures"]
            and not commitments["roles"]["AGG_CALIBRATION"]["released_in_r0"]
        ),
        "validation_unseen": (
            role_counts["AGG_VALIDATION"] == expected["sealed_AGG_VALIDATION_closures"]
            == resources["untouched"]["validation_closures"]
            and not commitments["roles"]["AGG_VALIDATION"]["released_in_r0"]
        ),
        "reserve_unseen": (
            role_counts["PUBLIC_RESERVE"] == expected["sealed_PUBLIC_RESERVE_closures"]
            == resources["untouched"]["public_reserve_closures"]
            and not commitments["roles"]["PUBLIC_RESERVE"]["released_in_r0"]
        ),
        "protected_unseen": (
            resources["untouched"]["protected_qualification_closures"] == 128
            and resources["untouched"]["protected_confirmation_closures"] == 256
        ),
        "prospective_roles_reconcile": sum(
            role["closures"] for role in allocation["prospective_rng2_roles"]
        ) == 1920,
    }
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_s0.capacity.v1",
        "gates": gates,
        "capacity_ready_for_formal_migration": all(gates.values()),
        "fresh_unallocated_closures": 0,
        "released_development_closures": 256,
        "unseen_public_closures": 1664,
        "unseen_protected_closures": 384,
        "role_identities_revealed_by_S0": 0,
        "role_migrations_performed_by_S0": 0,
    }


def evaluate() -> dict[str, Any]:
    protocol = validate_protocol()
    vectors = static_vectors()
    capacity = capacity_census()
    source_schedule = source_schedule_census()
    gates = {
        **protocol["gates"],
        "source_schedule": (
            source_schedule["sample_ids_call_sites_in_generate"] == 2
            and source_schedule["field_sequence_lengths"] == [12]
            and source_schedule["derived_runtime_draws"] == 49
            and source_schedule["none_appended_to_model_history"]
            and source_schedule["none_excluded_from_public_tokens"]
            and source_schedule["history_conditioned_predecessor_uniform"]
        ),
        "static_vectors": (
            vectors["paired_equal"]
            and vectors["different_context_separated"]
            and vectors["cdf_fixtures"]["u_boundary"]["selected_index"] == 0
            and vectors["cdf_fixtures"]["u_above_boundary"]["selected_index"] == 1
        ),
        "capacity": capacity["capacity_ready_for_formal_migration"],
        "zero_access": (
            capacity["role_identities_revealed_by_S0"] == 0
            and capacity["role_migrations_performed_by_S0"] == 0
        ),
    }
    ready = all(gates.values())
    config = protocol["config"]
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_s0.official.v1",
        "gates": gates,
        "static_vectors": vectors,
        "capacity": capacity,
        "source_schedule": source_schedule,
        "new_model_generations": 0,
        "new_stochastic_outputs": 0,
        "new_detector_scores": 0,
        "sealed_identity_releases": 0,
        "protected_releases": 0,
        "RNG2_R0_authorized": ready,
        "RNG2_D0_authorized": False,
        "protected_release_authorized": False,
        "passed": ready,
        "terminal": (
            config["terminals"]["ready"]
            if ready else config["terminals"]["implementation_invalid"]
        ),
    }


def validate_source_commit(source_commit: str) -> dict[str, str]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("SOURCE_COMMIT")
    result = {}
    for relative in SOURCE_FILES:
        local = (ROOT / relative).read_bytes()
        committed = subprocess.check_output(
            ["git", "show", source_commit + ":" + relative], cwd=ROOT
        )
        if committed != local:
            raise ValueError("SOURCE_BYTES:" + relative)
        result[relative] = sha(local)
    return result


def materialize(source_commit: str, output: Path) -> None:
    if output.exists():
        raise ValueError("OUTPUT_BOUNDARY")
    source_digests = validate_source_commit(source_commit)
    result = evaluate()
    output.mkdir(parents=True)
    vectors = result.pop("static_vectors")
    capacity = result.pop("capacity")
    source_schedule = result.pop("source_schedule")
    result["source_commit"] = source_commit
    result["source_digests"] = source_digests
    write_json(output / "STATIC_TEST_VECTORS.json", vectors)
    write_json(output / "CAPACITY_CENSUS.json", capacity)
    write_json(output / "SOURCE_SCHEDULE_CENSUS.json", source_schedule)
    write_json(output / "OFFICIAL_RESULT.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        print((canonical(evaluate()) + b"\n").decode(), end="")
    elif args.source_commit and args.output:
        materialize(args.source_commit, args.output)
    else:
        parser.error("use --evaluate or --source-commit with --output")


if __name__ == "__main__":
    main()
