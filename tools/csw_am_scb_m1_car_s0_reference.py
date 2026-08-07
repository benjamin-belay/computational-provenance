#!/usr/bin/env python3
"""Independent static evaluator for the SCB-M1 receipt-bound carrier protocol."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "protocol/csw-am-scb-m1-car-s0"


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in items:
        if key in out:
            raise ValueError("DUPLICATE:" + key)
        out[key] = value
    return out


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT")
    return value


def enc(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def value_hash(value: Any) -> str:
    return sha_bytes(enc(value))


def derive(root: bytes, category: int, spec: dict[str, Any]) -> bytes:
    projection = dict(spec["verified_claim_projection"]["fixed_values"])
    projection["category"] = category
    return hmac.new(
        root,
        spec["derivation"]["domain"].encode() + b"\x00" + enc(projection),
        hashlib.sha256,
    ).digest()


def evaluate(source_commit: str, package_path: Path, output: Path) -> dict[str, Any]:
    config = load(P / "CONFIG.json")
    authority = load(P / "AUTHORITY_PROTOCOL.json")
    boundary = load(P / "REALISER_BOUNDARY.json")
    design = load(P / "EXPERIMENT_DESIGN.json")
    package = load(package_path)
    source_ok = all(
        subprocess.check_output(
            ["git", "show", f"{source_commit}:{relative}"], cwd=ROOT
        ) == (ROOT / relative).read_bytes()
        for relative in config["source_file_paths"]
    )
    protocol_hashes_ok = all(
        sha_bytes((ROOT / relative).read_bytes()) == expected
        for relative, expected in package["protocol_files"].items()
    )
    receipt = config["receipt_binding"]
    receipt_audit = load(ROOT / receipt["main_audit_path"])
    receipt_protocol = load(ROOT / receipt["receipt_protocol_path"])
    receipt_ok = (
        sha_bytes((ROOT / receipt["main_audit_path"]).read_bytes())
        == receipt["main_audit_sha256"]
        and receipt_audit["main_verdict"] == receipt["main_terminal"]
        and sha_bytes((ROOT / receipt["receipt_protocol_path"]).read_bytes())
        == receipt["receipt_protocol_sha256"]
        and receipt_protocol["state_ontology"]["identity"] == receipt["state_ontology_id"]
        and receipt_protocol["state_ontology"]["z2_categories"] == 16
    )
    channel = config["channel_binding"]
    frozen = load(ROOT / channel["frozen_channel_path"])
    handoff = load(ROOT / channel["builder_handoff_path"])
    channel_ok = (
        sha_bytes((ROOT / channel["frozen_channel_path"]).read_bytes())
        == channel["frozen_channel_sha256"]
        and handoff["terminal"] == channel["builder_terminal"]
        and frozen["bias"] == 2.0
        and frozen["threshold"] == 20
        and frozen["channel"]["eligible_positions"] == 24
        and frozen["channel"]["alternatives_per_position"] == 8
        and frozen["authority_cardinality"] == 8
        and frozen["channel"]["authority_cardinality"] == 8
        and frozen["green_fraction"] == 0.5
        and sha_bytes((ROOT / channel["production_path"]).read_bytes())
        == channel["production_sha256"]
        and sha_bytes((ROOT / channel["reference_path"]).read_bytes())
        == channel["reference_sha256"]
    )
    root = bytes(range(32))
    authorities = [derive(root, category, authority) for category in range(16)]
    fixture = package["fixture"]
    authority_ok = (
        len(set(authorities)) == 16
        and fixture["authority_digests"] == [value.hex() for value in authorities]
        and fixture["same_state_different_exact_authority_equal"]
        and fixture["alias_offset_8_authority_different"]
        and fixture["invalid_receipt_no_authority"]
        and fixture["missing_receipt_no_authority"]
    )
    q = design["qualification_denominators"]
    c = design["confirmation_denominators"]
    o = design["output_design_per_closure"]
    d = design["detector_design_per_closure"]
    gates = {
        "source_binding": source_ok and package["source_commit"] == source_commit,
        "receipt_binding": receipt_ok,
        "channel_binding": channel_ok,
        "channel_parameters_frozen": channel["retuned_parameters"] == [],
        "all16_change_declared": (
            channel["qualified_candidate_multiplicity"] == 8
            and channel["integrated_candidate_multiplicity"] == 16
            and design["all16_integration_boundary"]["all16_familywise_error_inherited_from_A0_Q1"] is False
            and design["all16_integration_boundary"]["all16_familywise_error_must_pass_CAR_Q1"] is True
        ),
        "authority_projection_closed": (
            authority["verified_claim_projection"]["fields"]
            == list(authority["verified_claim_projection"]["fixed_values"]) + ["category"]
            and authority["authority_root"]["separate_from_receipt_mac_keys"]
            and "exact_receipt_id" in authority["verified_claim_projection"]["excluded_fields"]
        ),
        "authority_equivalence_fixture": authority_ok,
        "realiser_information_boundary": (
            set(boundary["allowed_inputs"]) == {
                "fixed_semantic_payload", "carrier_authority",
                "sampling_nonce", "frozen_generation_configuration",
            }
            and boundary["required_invariants"]["only_state_authority_may_switch"]
            and "natural_or_alias_label" in boundary["prohibited_inputs"]
        ),
        "paired_payload_and_nonce_fixture": (
            fixture["natural_alias_payload_bytes_equal"]
            and fixture["payload_answer_preserved"]
            and fixture["paired_nonce_equal"]
            and fixture["different_draw_nonce_different"]
        ),
        "qualification_denominators": (
            q["generated_outputs"] == q["closures"] * o["total_outputs"]
            and q["detector_decisions"] == q["closures"] * d["total_detector_decisions"]
            and q["eligible_positions"] == q["generated_outputs"] * 24
        ),
        "confirmation_denominators": (
            c["generated_outputs"] == c["closures"] * o["total_outputs"]
            and c["detector_decisions"] == c["closures"] * d["total_detector_decisions"]
            and c["eligible_positions"] == c["generated_outputs"] * 24
        ),
        "closure_level_gates": (
            design["primary_unit"] == "DEPENDENCY_CLOSURE"
            and design["statistical_gates"]["closure_pair_switch_success_fraction_min"] > 0
            and design["statistical_gates"]["no_gate_compensation"]
        ),
        "commit_reveal_lifecycle": (
            design["lifecycle"]["commit_all_raw_receipts_outputs_and_traces_before_key_reveal"]
            and design["lifecycle"]["no_precommit_output_filtering"]
            and design["lifecycle"]["replacement_closures"] == 0
        ),
        "protected_roles_bound_and_sealed": (
            config["protected_roles"]["CARRIER_EXCLUDED_QUALIFICATION"]["closures"] == 128
            and config["protected_roles"]["CARRIER_SEALED_CONFIRMATION"]["closures"] == 256
            and not config["authorizations"]["CARRIER_EXCLUDED_QUALIFICATION_release"]
            and not config["authorizations"]["CARRIER_SEALED_CONFIRMATION_release"]
        ),
        "zero_scientific_work": all(
            value == 0 for value in package["zero_scientific_work"].values()
        ),
        "public_D0_only": (
            config["authorizations"]["public_CAR_D0_after_valid_S0"]
            and not config["authorizations"]["automatic_successor"]
        ),
    }
    rows = [
        {"gate_id": key, "status": "PASS" if value else "FAIL"}
        for key, value in sorted(gates.items())
    ]
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_s0.reference_result.v1",
        "evaluator": "REFERENCE",
        "imports_official_logic": False,
        "protocol_hashes_valid": protocol_hashes_ok,
        "gates": rows,
        "gate_digest": value_hash(rows),
        "passed": all(row["status"] == "PASS" for row in rows) and protocol_hashes_ok,
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(enc(result) + b"\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.source_commit, args.package, args.output)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
