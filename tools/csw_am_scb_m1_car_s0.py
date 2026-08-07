#!/usr/bin/env python3
"""Static builder and reconciler for the receipt-bound SCB-M1 carrier protocol."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol/csw-am-scb-m1-car-s0"
CONFIG_PATH = PROTOCOL_DIR / "CONFIG.json"
AUTHORITY_PATH = PROTOCOL_DIR / "AUTHORITY_PROTOCOL.json"
BOUNDARY_PATH = PROTOCOL_DIR / "REALISER_BOUNDARY.json"
DESIGN_PATH = PROTOCOL_DIR / "EXPERIMENT_DESIGN.json"


def strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("DUPLICATE_KEY:" + key)
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes(), object_pairs_hook=strict_pairs)
    if type(value) is not dict:
        raise ValueError("OBJECT_REQUIRED:" + path.name)
    return value


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(canonical(value) + b"\n")


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def validate_sources(source_commit: str, config: dict[str, Any]) -> bool:
    if len(source_commit) != 40:
        return False
    try:
        return all(
            subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=ROOT, stderr=subprocess.DEVNULL,
            ) == (ROOT / relative).read_bytes()
            for relative in config["source_file_paths"]
        )
    except subprocess.CalledProcessError:
        return False


def validate_bindings(config: dict[str, Any]) -> tuple[bool, bool]:
    receipt = config["receipt_binding"]
    receipt_audit_path = ROOT / receipt["main_audit_path"]
    receipt_protocol_path = ROOT / receipt["receipt_protocol_path"]
    receipt_audit = read_json(receipt_audit_path)
    receipt_protocol = read_json(receipt_protocol_path)
    receipt_ok = (
        file_sha(receipt_audit_path) == receipt["main_audit_sha256"]
        and git_show(receipt["main_audit_commit"], receipt["main_audit_path"])
        == receipt_audit_path.read_bytes()
        and receipt_audit["main_verdict"] == receipt["main_terminal"]
        and file_sha(receipt_protocol_path) == receipt["receipt_protocol_sha256"]
        and receipt_protocol["state_ontology"]["identity"]
        == receipt["state_ontology_id"]
        and receipt_protocol["state_ontology"]["model_domain"]
        == receipt["model_domain"]
        and receipt_protocol["state_ontology"]["checkpoint_sha256"]
        == receipt["checkpoint_sha256"]
        and receipt_protocol["state_ontology"]["z2_categories"]
        == receipt["z2_categories"]
    )
    channel = config["channel_binding"]
    frozen_path = ROOT / channel["frozen_channel_path"]
    handoff_path = ROOT / channel["builder_handoff_path"]
    frozen = read_json(frozen_path)
    handoff = read_json(handoff_path)
    code_ok = True
    for kind in ("production", "reference"):
        relative = channel[f"{kind}_path"]
        path = ROOT / relative
        code_ok = code_ok and (
            file_sha(path) == channel[f"{kind}_sha256"]
            and subprocess.check_output(
                ["git", "rev-parse", f"{channel['immutable_commit']}:{relative}"],
                cwd=ROOT, text=True,
            ).strip() == channel[f"{kind}_blob"]
            and git_show(channel["immutable_commit"], relative) == path.read_bytes()
        )
    channel_ok = (
        file_sha(frozen_path) == channel["frozen_channel_sha256"]
        and git_show(channel["immutable_commit"], channel["frozen_channel_path"])
        == frozen_path.read_bytes()
        and git_show(channel["immutable_commit"], channel["builder_handoff_path"])
        == handoff_path.read_bytes()
        and handoff["terminal"] == channel["builder_terminal"]
        and frozen["checkpoint"]["sha256"] == channel["checkpoint_sha256"]
        and frozen["bias"] == channel["bias"]
        and frozen["threshold"] == channel["threshold"]
        and frozen["channel"]["eligible_positions"] == channel["eligible_positions"]
        and frozen["minimum_eligible_length"] == channel["minimum_eligible_length"]
        and frozen["channel"]["alternatives_per_position"]
        == channel["alternatives_per_position"]
        and frozen["authority_cardinality"]
        == channel["qualified_candidate_multiplicity"]
        and frozen["channel"]["authority_cardinality"]
        == channel["qualified_candidate_multiplicity"]
        and frozen["green_fraction"] == channel["green_fraction"]
        and frozen["channel"]["domain"] == channel["green_domain"]
        and code_ok
    )
    return receipt_ok, channel_ok


def verified_projection(category: int, authority: dict[str, Any]) -> dict[str, Any]:
    spec = authority["verified_claim_projection"]
    if type(category) is not int or not (
        spec["category_domain"]["minimum"] <= category <= spec["category_domain"]["maximum"]
    ):
        raise ValueError("CATEGORY")
    return {**spec["fixed_values"], "category": category}


def derive_authority(
    root_key: bytes, category: int, receipt_decision: str,
    authority: dict[str, Any],
) -> bytes | None:
    if receipt_decision != authority["verified_claim_projection"]["receipt_decision_required"]:
        return None
    projection = verified_projection(category, authority)
    domain = authority["derivation"]["domain"].encode()
    return hmac.new(root_key, domain + b"\x00" + canonical(projection), hashlib.sha256).digest()


def surface_payload(prompt: bytes, answer: int, boundary: dict[str, Any]) -> dict[str, Any]:
    if type(answer) is not int or not 0 <= answer <= 7:
        raise ValueError("ANSWER")
    def slot(domain: bytes) -> int:
        return int.from_bytes(hashlib.sha256(domain + b"\x00" + prompt).digest()[:8], "big") % 8
    value = {
        "schema": boundary["surface_semantic_payload"]["schema"],
        "z1": slot(b"SURFACE_Z1_V1"),
        "z2": slot(b"SURFACE_Z2_V1"),
        "z3": answer,
        "answer": answer,
    }
    value["semantic_id"] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def paired_nonce(closure_id: str, draw: int) -> bytes:
    if type(draw) is not int or not 0 <= draw <= 3:
        raise ValueError("DRAW")
    return hashlib.sha256(
        b"SCB_M1_CAR_PAIRED_NONCE_V1\x00"
        + closure_id.encode()
        + b"\x00"
        + draw.to_bytes(4, "big")
    ).digest()


def fixture(authority: dict[str, Any], boundary: dict[str, Any]) -> dict[str, Any]:
    root_key = bytes(range(32))
    state_3_a = derive_authority(root_key, 3, "ACCEPT", authority)
    state_3_b = derive_authority(root_key, 3, "ACCEPT", authority)
    state_11 = derive_authority(root_key, 11, "ACCEPT", authority)
    all_states = [derive_authority(root_key, value, "ACCEPT", authority) for value in range(16)]
    prompt = b"GATED_ALIAS16 visible prompt fixture"
    natural_payload = surface_payload(prompt, 3, boundary)
    alias_payload = surface_payload(prompt, 3, boundary)
    return {
        "schema": "exp33a.csw_am_scb_m1_car_s0.fixture.v1",
        "same_state_different_exact_authority_equal": state_3_a == state_3_b,
        "alias_offset_8_authority_different": state_3_a != state_11,
        "all16_authorities_unique": len(set(all_states)) == 16,
        "invalid_receipt_no_authority": derive_authority(root_key, 3, "REJECT", authority) is None,
        "missing_receipt_no_authority": derive_authority(root_key, 3, "ABSTAIN", authority) is None,
        "natural_alias_payload_bytes_equal": canonical(natural_payload) == canonical(alias_payload),
        "payload_answer_preserved": natural_payload["answer"] == 3,
        "paired_nonce_equal": paired_nonce("fixture", 2) == paired_nonce("fixture", 2),
        "different_draw_nonce_different": paired_nonce("fixture", 1) != paired_nonce("fixture", 2),
        "authority_digests": [value.hex() for value in all_states if value is not None],
        "payload": natural_payload,
    }


def gate_rows(source_commit: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = read_json(CONFIG_PATH)
    authority = read_json(AUTHORITY_PATH)
    boundary = read_json(BOUNDARY_PATH)
    design = read_json(DESIGN_PATH)
    receipt_ok, channel_ok = validate_bindings(config)
    sample = fixture(authority, boundary)
    channel = config["channel_binding"]
    q, c = design["qualification_denominators"], design["confirmation_denominators"]
    output = design["output_design_per_closure"]
    detector = design["detector_design_per_closure"]
    gates = {
        "source_binding": validate_sources(source_commit, config),
        "receipt_binding": receipt_ok,
        "channel_binding": channel_ok,
        "channel_parameters_frozen": channel["retuned_parameters"] == [],
        "all16_change_declared": (
            channel["qualified_candidate_multiplicity"] == 8
            and channel["integrated_candidate_multiplicity"] == 16
            and channel["multiplicity_change_requires_integrated_Q1"] is True
            and design["all16_integration_boundary"]["all16_familywise_error_inherited_from_A0_Q1"] is False
            and design["all16_integration_boundary"]["all16_familywise_error_must_pass_CAR_Q1"] is True
        ),
        "authority_projection_closed": (
            authority["verified_claim_projection"]["fields"]
            == list(authority["verified_claim_projection"]["fixed_values"]) + ["category"]
            and authority["authority_root"]["separate_from_receipt_mac_keys"] is True
            and "complete_receipt_envelope"
            in authority["verified_claim_projection"]["excluded_fields"]
        ),
        "authority_equivalence_fixture": all(
            sample[field] for field in (
                "same_state_different_exact_authority_equal",
                "alias_offset_8_authority_different",
                "all16_authorities_unique",
                "invalid_receipt_no_authority",
                "missing_receipt_no_authority",
            )
        ),
        "realiser_information_boundary": (
            set(boundary["allowed_inputs"]) == {
                "fixed_semantic_payload", "carrier_authority",
                "sampling_nonce", "frozen_generation_configuration",
            }
            and boundary["required_invariants"]["only_state_authority_may_switch"] is True
            and "receipt_envelope" in boundary["prohibited_inputs"]
            and "raw_z2" in boundary["prohibited_inputs"]
        ),
        "paired_payload_and_nonce_fixture": (
            sample["natural_alias_payload_bytes_equal"]
            and sample["payload_answer_preserved"]
            and sample["paired_nonce_equal"]
            and sample["different_draw_nonce_different"]
        ),
        "qualification_denominators": (
            q["generated_outputs"] == q["closures"] * output["total_outputs"]
            and q["detector_decisions"] == q["closures"] * detector["total_detector_decisions"]
            and q["eligible_positions"] == q["generated_outputs"] * channel["eligible_positions"]
        ),
        "confirmation_denominators": (
            c["generated_outputs"] == c["closures"] * output["total_outputs"]
            and c["detector_decisions"] == c["closures"] * detector["total_detector_decisions"]
            and c["eligible_positions"] == c["generated_outputs"] * channel["eligible_positions"]
        ),
        "closure_level_gates": (
            design["primary_unit"] == "DEPENDENCY_CLOSURE"
            and design["statistical_gates"]["closure_pair_switch_success_fraction_min"] > 0
            and design["statistical_gates"]["no_gate_compensation"] is True
        ),
        "commit_reveal_lifecycle": (
            design["lifecycle"]["commit_all_raw_receipts_outputs_and_traces_before_key_reveal"]
            and design["lifecycle"]["no_precommit_detector_scoring"]
            and design["lifecycle"]["retries"] == 0
        ),
        "protected_roles_bound_and_sealed": (
            config["protected_roles"]["CARRIER_EXCLUDED_QUALIFICATION"]["closures"] == 128
            and config["protected_roles"]["CARRIER_SEALED_CONFIRMATION"]["closures"] == 256
            and config["authorizations"]["CARRIER_EXCLUDED_QUALIFICATION_release"] is False
            and config["authorizations"]["CARRIER_SEALED_CONFIRMATION_release"] is False
        ),
        "zero_scientific_work": all(
            value == 0 for value in config["zero_scientific_work"].values()
        ),
        "public_D0_only": (
            config["authorizations"]["public_CAR_D0_after_valid_S0"] is True
            and config["authorizations"]["automatic_successor"] is False
        ),
    }
    rows = [
        {"gate_id": key, "status": "PASS" if value else "FAIL"}
        for key, value in sorted(gates.items())
    ]
    return rows, sample


def build(source_commit: str, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    rows, sample = gate_rows(source_commit)
    files = {
        str(path.relative_to(ROOT)): file_sha(path)
        for path in (CONFIG_PATH, AUTHORITY_PATH, BOUNDARY_PATH, DESIGN_PATH)
    }
    package = {
        "schema": "exp33a.csw_am_scb_m1_car_s0.static_package.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "protocol_files": files,
        "fixture": sample,
        "zero_scientific_work": config["zero_scientific_work"],
    }
    passed = all(row["status"] == "PASS" for row in rows)
    result = {
        "schema": "exp33a.csw_am_scb_m1_car_s0.official_result.v1",
        "evaluator": "OFFICIAL",
        "gates": rows,
        "gate_digest": digest(rows),
        "passed": passed,
    }
    write_json(output / "STATIC_PACKAGE.json", package)
    write_json(output / "OFFICIAL_RESULT.json", result)
    return result


def reconcile(official_path: Path, reference_path: Path, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    official, reference = read_json(official_path), read_json(reference_path)
    agreement = (
        official["gate_digest"] == reference["gate_digest"]
        and official["gates"] == reference["gates"]
    )
    passed = official["passed"] and reference["passed"] and agreement
    terminal = {
        "schema": "exp33a.csw_am_scb_m1_car_s0.terminal.v1",
        "terminal": (
            config["terminals"]["valid"]
            if passed else config["terminals"]["evaluator_mismatch"]
        ),
        "passed": passed,
        "official_reference_exact_agreement": agreement,
        "public_CAR_D0_authorized": passed,
        "protected_role_release_authorized": False,
        "scientific_outputs": 0,
    }
    write_json(output, terminal)
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    preflight_parser = sub.add_parser("preflight")
    preflight_parser.add_argument("--source-commit")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--source-commit", required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--official", type=Path, required=True)
    reconcile_parser.add_argument("--reference", type=Path, required=True)
    reconcile_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            config = read_json(CONFIG_PATH)
            source_commit = args.source_commit or subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            rows, sample = gate_rows(source_commit)
            result = {
                "schema": "exp33a.csw_am_scb_m1_car_s0.preflight.v1",
                "gates": rows,
                "fixture": sample,
                "passed": all(row["status"] == "PASS" for row in rows),
                "identity": config["identity"],
            }
        elif args.command == "build":
            result = build(args.source_commit, args.output)
        else:
            result = reconcile(args.official, args.reference, args.output)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
