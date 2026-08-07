#!/usr/bin/env python3
"""Independent static reconstruction of the RNG2-S0 protocol."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DOMAIN = b"EXP33A|SCB-M1|RNG2|POSITION-UNIFORM|V1"


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


def independent_uniform(key: bytes, index: dict[str, Any]) -> tuple[float, str]:
    digest = hmac.new(key, DOMAIN + b"\x00" + canonical(index), hashlib.sha256).digest()
    integer = int.from_bytes(digest[:8], "big") // (1 << 12)
    return (integer + 0.5) / (1 << 52), digest.hex()


def assess(evidence: Path) -> dict[str, Any]:
    config = read(ROOT / "protocol/csw-am-scb-m1-rng2-s0/CONFIG.json")
    sampler = read(ROOT / "protocol/csw-am-scb-m1-rng2-s0/SAMPLER_PROTOCOL.json")
    numeric = read(
        ROOT / "protocol/csw-am-scb-m1-rng2-s0/NUMERIC_AND_SCHEDULE_PROTOCOL.json"
    )
    validation = read(ROOT / "protocol/csw-am-scb-m1-rng2-s0/VALIDATION_PROTOCOL.json")
    allocation = read(ROOT / "protocol/csw-am-scb-m1-rng2-s0/ALLOCATION_PROTOCOL.json")
    official = read(evidence / "OFFICIAL_RESULT.json")
    vectors = read(evidence / "STATIC_TEST_VECTORS.json")
    capacity = read(evidence / "CAPACITY_CENSUS.json")
    source_schedule = read(evidence / "SOURCE_SCHEDULE_CENSUS.json")
    uniform, digest = independent_uniform(bytes.fromhex(vectors["key_hex"]), vectors["canonical_index"])
    gates = {
        "official_passed": official["passed"] is True,
        "terminal": official["terminal"]
        == "MAIN_CSW_AM_SCB_M1_RNG2_S0_PROTOCOL_READY_FOR_ROLE_MIGRATION",
        "uniform_reconstructed": (
            uniform == vectors["uniform"] and digest == vectors["prf_digest"]
        ),
        "history_excluded": "generated_history"
        in sampler["canonical_index"]["prohibited_fields"],
        "state_and_role_excluded": all(
            item in sampler["canonical_index"]["prohibited_fields"]
            for item in ("natural_or_alias_role", "z2", "z3")
        ),
        "pairing_context_not_exact_execution": (
            sampler["pairing"]["pairing_context_is_distinct_from_exact_run_and_execution_identity"]
        ),
        "fixed_schedule": (
            numeric["draw_schedule"]["total_model_steps"] == 49
            and numeric["draw_schedule"]["subdraws_per_step"] == 1
            and numeric["none_and_termination"]["NONE_does_not_terminate_generation"]
            and source_schedule["derived_runtime_draws"] == 49
            and source_schedule["field_sequence_lengths"] == [12]
        ),
        "new_channel_requalification": not any(
            validation["generic_channel_requalification"][key]
            for key in (
                "A0_Q1_power_inherited", "A0_Q1_semantic_guarantee_inherited",
                "A0_Q1_false_positive_guarantee_inherited",
            )
        ),
        "migration_required": (
            allocation["status"] == "FORMAL_MIGRATION_REQUIRED_BEFORE_ANY_RNG2_GENERATION"
            and capacity["fresh_unallocated_closures"] == 0
            and capacity["role_migrations_performed_by_S0"] == 0
        ),
        "no_empirical_work": (
            official["new_model_generations"] == 0
            and official["new_stochastic_outputs"] == 0
            and official["new_detector_scores"] == 0
            and official["sealed_identity_releases"] == 0
            and official["protected_releases"] == 0
        ),
        "only_r0_authorized": (
            official["RNG2_R0_authorized"] is True
            and official["RNG2_D0_authorized"] is False
            and official["protected_release_authorized"] is False
            and config["authorizations"]["automatic_successor"] is False
        ),
    }
    valid = all(gates.values())
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_s0.reference.v1",
        "gates": gates,
        "valid": valid,
        "uniform": uniform,
        "prf_digest": digest,
        "terminal": (
            "MAIN_CSW_AM_SCB_M1_RNG2_S0_PROTOCOL_READY_FOR_ROLE_MIGRATION"
            if valid else "MAIN_CSW_AM_SCB_M1_RNG2_S0_IMPLEMENTATION_INVALID"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = assess(args.evidence)
    data = canonical(result) + b"\n"
    if args.output:
        if args.output.exists():
            raise ValueError("OUTPUT_BOUNDARY")
        args.output.write_bytes(data)
    else:
        print(data.decode(), end="")


if __name__ == "__main__":
    main()
