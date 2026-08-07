#!/usr/bin/env python3
"""Static builder for the RNG2 receipt-bound integration protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "protocol/csw-am-scb-m1-rng2-int-s0"
CONFIG = BASE / "CONFIG.json"
PROTOCOL = BASE / "INTEGRATION_PROTOCOL.json"
MATRIX = BASE / "VERDICT_MATRIX.json"
ROLE = BASE / "PUBLIC_ROLE_BINDING.json"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def committed(path: Path, commit: str) -> bool:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{path.relative_to(ROOT)}"],
            cwd=ROOT, stderr=subprocess.DEVNULL,
        ) == path.read_bytes()
    except subprocess.CalledProcessError:
        return False


def build(source_commit: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("STATIC_BOUNDARY")
    config, protocol, matrix, role = map(read_json, (CONFIG, PROTOCOL, MATRIX, ROLE))
    receipt_audit = read_json(ROOT / config["bindings"]["receipt_main_path"])
    receipt_terminal = read_json(ROOT / config["bindings"]["receipt_terminal_path"])
    receipt_protocol = read_json(ROOT / config["bindings"]["receipt_protocol_path"])
    rng2 = read_json(ROOT / config["bindings"]["rng2_v1_main_path"])
    rng2_terminal = read_json(ROOT / config["bindings"]["rng2_v1_terminal_path"])
    reserve = read_json(ROOT / config["bindings"]["reserve_commitments_path"])

    source_hashes = {
        name: (
            file_sha(ROOT / config["bindings"][name + "_path"])
            == config["bindings"][name + "_sha256"]
        )
        for name in (
            "receipt_main", "receipt_terminal", "receipt_protocol",
            "rng2_v1_main", "rng2_v1_terminal", "rng2_v1_official",
            "reserve_commitments", "reserve_main", "reserve_terminal",
        )
    }
    gates = {
        "source_hashes": all(source_hashes.values()),
        "source_commit": (
            len(source_commit) == 40
            and all(committed(ROOT / relative, source_commit) for relative in config["source_paths"])
        ),
        "receipt_confirmation": (
            receipt_audit["main_verdict"] == config["bindings"]["receipt_terminal"]
            and receipt_terminal["terminal"]
            == "CSW_AM_SCB_M1_REC_C1_RECEIPT_PROVENANCE_CONFIRMED"
            and receipt_terminal["receipt_to_carrier_protocol_authorized"] is True
        ),
        "state_ontology": (
            receipt_protocol["state_ontology"]["identity"]
            == config["state_ontology"]["identity"]
            and receipt_protocol["state_ontology"]["model_domain"]
            == config["state_ontology"]["model_domain"]
            and receipt_protocol["state_ontology"]["checkpoint_sha256"]
            == config["state_ontology"]["checkpoint_sha256"]
            and receipt_protocol["state_ontology"]["z2_categories"]
            == config["state_ontology"]["categories"]
        ),
        "generic_channel": (
            rng2["main_verdict"] == config["bindings"]["rng2_v1_terminal"]
            and rng2["scientific_pass"] is True
            and rng2_terminal["terminal"] == config["bindings"]["rng2_v1_terminal"]
            and rng2["thresholds"]["T"] == config["frozen_channel"]["T"]
            and rng2["thresholds"]["M"] == config["frozen_channel"]["M"]
        ),
        "receipt_before_authority": protocol["claim_bearing_generation_order"][:4] == [
            "VERIFY_EXACT_Z2_RECEIPT",
            "VERIFY_ABSTRACT_Z2_RECEIPT",
            "VERIFY_CAUSAL_CHAIN",
            "EXTRACT_AUTHENTICATED_STATE_CLASS_FIELDS",
        ],
        "authority_projection": (
            protocol["authority_preimage"]["same_state_different_execution_same_authority"] is True
            and protocol["authority_preimage"]["different_state_different_authority"] is True
            and set(protocol["authority_preimage"]["excluded_fields"]) >= {
                "receipt_MAC_bytes", "exact_receipt_id", "exact_execution_id",
                "nonce", "natural_alias_role", "prompt", "answer", "generated_history",
            }
        ),
        "separate_verdicts": (
            matrix["channels"] == [
                "exact_event_receipt", "abstract_state_receipt", "causal_chain",
                "statistical_state_carrier", "integrated_provenance",
            ]
            and len(matrix["cases"]) >= 6
        ),
        "channel_unchanged": (
            protocol["unchanged_channel"]["T"] == rng2["thresholds"]["T"]
            and protocol["unchanged_channel"]["M"] == rng2["thresholds"]["M"]
            and protocol["unchanged_channel"]["outputs_per_side"] == 8
            and protocol["unchanged_channel"]["authorities"] == 16
            and protocol["unchanged_channel"]["steps"] == 49
            and protocol["unchanged_channel"]["fallback"] is False
            and config["frozen_channel"]["statistical_changes_authorized"] is False
        ),
        "role_membership_preserved": (
            reserve["roles"]["RNG2_REMAINING_PUBLIC_RESERVE"]["closures"] == 128
            and reserve["roles"]["RNG2_REMAINING_PUBLIC_RESERVE"]["merkle_root"]
            == role["source_membership_root"] == role["target_membership_root"]
            and role["membership_preserved"] is True
            and role["redraw"] is False
            and role["identity_release"] == 0
        ),
        "error_interpretation": (
            config["error_interpretation"]["generic_one_percent_control_inherited"] is True
            and config["error_interpretation"]["inheritance_requires_byte_identical_channel"] is True
            and config["error_interpretation"]["INT_V1_maximum_carrier_off_events"] == 0
            and config["error_interpretation"]["INT_V1_maximum_wrong_confusion_events"] == 0
            and config["error_interpretation"]["INT_V1_independently_reestimates_one_percent_rate"] is False
            and math.isclose(
                config["error_interpretation"]["zero_event_one_sided_95_upper_bound"],
                1.0 - math.pow(0.05, 1.0 / 128.0),
            )
        ),
        "static_zero_work": (
            config["authorizations"]["generation"] is False
            and config["authorizations"]["detector_scoring"] is False
            and config["authorizations"]["INT_V1_release"] is False
            and config["authorizations"]["reserve_identity_release"] is False
            and config["authorizations"]["protected_release"] is False
        ),
    }
    passed = all(gates.values())
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_s0.official.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "gates": gates,
        "source_hashes": source_hashes,
        "passed": passed,
        "terminal": (
            config["terminals"]["ready"] if passed
            else config["terminals"]["protocol_invalid"]
        ),
        "frozen_channel": config["frozen_channel"],
        "public_roles": config["public_roles"],
        "error_interpretation": config["error_interpretation"],
        "zero_scientific_work": {
            "new_model_generations": 0,
            "detector_scores": 0,
            "receipts_issued": 0,
            "public_identity_releases": 0,
            "protected_releases": 0
        },
        "authorizations": {
            "INT_D0_public_implementation": passed,
            "INT_V1_release": False,
            "protected_release": False,
            "automatic_successor": False
        }
    }
    output.mkdir(parents=True)
    write_json(output / "OFFICIAL_RESULT.json", result)
    return result


def finalize(output: Path) -> dict[str, Any]:
    config = read_json(CONFIG)
    official = read_json(output / "OFFICIAL_RESULT.json")
    reference = read_json(output / "REFERENCE_RESULT.json")
    valid = official["passed"] is True and reference["passed"] is True
    audit = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_s0.main_audit.v1",
        "identity": config["identity"],
        "valid": valid,
        "main_verdict": (
            config["terminals"]["ready"] if valid
            else config["terminals"]["protocol_invalid"]
        ),
        "gates": official["gates"],
        "independent_agreement": reference["gates"],
        "role_binding": read_json(ROLE),
        "error_interpretation": config["error_interpretation"],
        "claim_boundary": {
            "composition_protocol_ready": valid,
            "receipt_bound_integration_implemented": False,
            "integrated_public_validation": False,
            "protected_output_provenance": False
        },
        "zero_scientific_work": official["zero_scientific_work"],
        "authorizations": official["authorizations"],
        "evidence": {
            "official_sha256": file_sha(output / "OFFICIAL_RESULT.json"),
            "reference_sha256": file_sha(output / "REFERENCE_RESULT.json")
        }
    }
    write_json(output / "MAIN_AUDIT.json", audit)
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_rng2_int_s0.terminal.v1",
        "terminal": audit["main_verdict"],
        "valid": valid,
        "main_audit_sha256": file_sha(output / "MAIN_AUDIT.json"),
        "INT_D0_authorized": valid,
        "INT_V1_release_authorized": False,
        "protected_release": 0
    }
    write_json(output / "TERMINAL.json", terminal)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source-commit", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    final_parser = sub.add_parser("finalize")
    final_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = (
        build(args.source_commit, args.output)
        if args.command == "build" else finalize(args.output)
    )
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
