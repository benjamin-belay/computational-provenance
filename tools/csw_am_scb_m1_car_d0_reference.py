#!/usr/bin/env python3
"""Independent reconstruction for the SCB Model-1 CAR-D0 public run."""
from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib.util
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-car-d0/CONFIG.json"
AUTHORITY_PATH = ROOT / "protocol/csw-am-scb-m1-car-s0/AUTHORITY_PROTOCOL.json"
RECEIPT_PROTOCOL_PATH = ROOT / "protocol/csw-am-scb-m1-rec-s0/RECEIPT_PROTOCOL.json"


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


REC_REF = load("car_d0_receipt_reference", "tools/csw_am_scb_m1_rec_d0_reference.py")
A0 = load("car_d0_frozen_surface", "tools/csw_a0_stochastic_channel.py")
A0_REF = load("car_d0_channel_reference", "tools/csw_a0_reference.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("OBJECT:" + path.name)
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def git_bytes(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def committed(path: Path, commit: str) -> None:
    relative = str(path.resolve().relative_to(ROOT))
    if git_bytes(commit, relative) != path.read_bytes():
        raise ValueError("COMMIT_BINDING:" + relative)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def state_projection(category: int, authority: dict[str, Any]) -> dict[str, Any]:
    spec = authority["verified_claim_projection"]
    if type(category) is not int or not (
        spec["category_domain"]["minimum"] <= category <= spec["category_domain"]["maximum"]
    ):
        raise ValueError("CATEGORY")
    return {**spec["fixed_values"], "category": category}


def state_authority(root: bytes, category: int, authority: dict[str, Any]) -> bytes:
    return hmac.new(
        root,
        authority["derivation"]["domain"].encode()
        + b"\x00"
        + canonical(state_projection(category, authority)),
        hashlib.sha256,
    ).digest()


def baseline_authority(root: bytes, family: str, prompt: str, answer: int) -> bytes:
    domain = {
        "PROMPT": "EXP33A|SCB-M1|PROMPT-CARRIER-AUTHORITY|V1",
        "ANSWER": "EXP33A|SCB-M1|ANSWER-CARRIER-AUTHORITY|V1",
        "GENERIC": "EXP33A|SCB-M1|GENERIC-CARRIER-AUTHORITY|V1",
    }[family]
    projection = {
        "PROMPT": {"visible_prompt_sha256": sha(prompt.encode())},
        "ANSWER": {"answer": answer},
        "GENERIC": {"constant_origin_label": "SCB_M1_GENERIC_ORIGIN_V1"},
    }[family]
    return hmac.new(
        root, domain.encode() + b"\x00" + canonical(projection), hashlib.sha256
    ).digest()


def paired_nonce(closure_id: str, draw: int) -> bytes:
    return hashlib.sha256(
        b"SCB_M1_CAR_PAIRED_NONCE_V1\x00"
        + closure_id.encode()
        + b"\x00"
        + draw.to_bytes(4, "big")
    ).digest()


def expected_binding(primitive: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": primitive["run_id"],
        "context_id": primitive["context_id"],
        "execution_id": primitive["execution_id"],
        "nonce": primitive["nonce"],
        "checkpoint_sha256": config["lineage"]["M1_checkpoint_sha256"],
    }


def key_commitment(key: bytes, domain: str) -> str:
    return sha(domain.encode() + b"\x00" + key)


def validate(
    source_commit: str, raw: Path, raw_commit: str,
    reveal_path: Path, reveal_commit: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    config = read_json(CONFIG_PATH)
    for relative in config["source_file_paths"]:
        if git_bytes(source_commit, relative) != (ROOT / relative).read_bytes():
            raise ValueError("SOURCE_BINDING:" + relative)
    committed(raw / "RAW_MANIFEST.json", raw_commit)
    committed(reveal_path, reveal_commit)
    manifest, reveal = read_json(raw / "RAW_MANIFEST.json"), read_json(reveal_path)
    if (
        reveal["raw_evidence_commit"] != raw_commit
        or reveal["raw_manifest_sha256"] != file_sha(raw / "RAW_MANIFEST.json")
    ):
        raise ValueError("REVEAL_BINDING")
    receipt = bytes.fromhex(reveal["receipt_key_hex"])
    carrier = bytes.fromhex(reveal["carrier_root_hex"])
    if (
        key_commitment(receipt, config["keys"]["receipt_commitment_domain"])
        != manifest["receipt_key_commitment"]
        or key_commitment(carrier, config["keys"]["carrier_commitment_domain"])
        != manifest["carrier_key_commitment"]
    ):
        raise ValueError("KEY_BINDING")
    for name, expected in manifest["files"].items():
        if file_sha(raw / name) != expected:
            raise ValueError("RAW_HASH:" + name)
    return config, receipt, carrier


def authority_for(
    output: dict[str, Any], root: bytes, authority: dict[str, Any],
    public: dict[str, dict[str, Any]],
) -> bytes | None:
    if output["condition"] == "STATE":
        return state_authority(root, output["source_z2_category"], authority)
    if output["condition"] == "OFF":
        return None
    row = public[output["closure_id"]]
    return baseline_authority(
        root, output["condition"], row["prompt_text"], row["natural"]["answer"]
    )


def gates(config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, bool]:
    expected, limits = config["execution"], config["gates"]
    return {
        "complete_denominator": (
            metrics["closures"] == config["public_population"]["selected_closures"]
            and metrics["execution_packages"] == expected["SCB_execution_packages"]
            and metrics["receipt_envelopes"] == expected["receipt_envelopes"]
            and metrics["generated_outputs"] == expected["generated_outputs"]
            and metrics["state_authority_scores"] == expected["state_authority_scores"]
            and metrics["own_family_control_scores"] == expected["own_family_control_scores"]
            and metrics["detector_decisions"] == expected["detector_decisions"]
            and metrics["eligible_positions"] == expected["eligible_positions"]
        ),
        "receipt_and_chain": metrics["receipt_and_chain_accepts"]
        == metrics["execution_packages"],
        "authority_equivalence": (
            metrics["all16_authorities_distinct"]
            and metrics["same_state_categories_with_multiple_exact_executions"] == 16
            and metrics["same_state_authority_equal_categories"] == 16
            and metrics["baseline_authority_equal"] == metrics["baseline_authority_total"]
            and metrics["state_authority_switches"] == metrics["closures"]
        ),
        "payload_equality": metrics["payload_pair_equal"] == metrics["payload_pair_total"],
        "paired_nonce_equality": metrics["paired_nonce_equal"]
        == metrics["paired_nonce_total"],
        "baseline_output_equality": metrics["baseline_output_equal"]
        == metrics["baseline_output_total"],
        "semantic_validity": metrics["semantic_successes"] == metrics["generated_outputs"],
        "generation_reconstruction": metrics["generation_reconstruction_matches"]
        == metrics["generated_outputs"],
        "matching_power": metrics["matching_state_accepts"] / metrics["state_outputs"]
        >= limits["matching_state_output_power_min"],
        "wrong_state_false_accept": (
            metrics["state_outputs_with_any_wrong_accept"] / metrics["state_outputs"]
            <= limits["state_output_any_wrong_state_false_accept_max"]
        ),
        "carrier_off_false_accept": (
            metrics["carrier_off_outputs_with_any_state_accept"]
            / metrics["carrier_off_outputs"]
            <= limits["carrier_off_any_state_false_accept_max"]
        ),
        "side_success": metrics["side_successes"] / metrics["sides"]
        >= limits["side_success_fraction_min"],
        "closure_pair_switch": (
            metrics["closure_pair_switch_successes"] / metrics["closures"]
            >= limits["closure_pair_switch_success_fraction_min"]
        ),
        "paired_margin": metrics["paired_margin_mean"]
        >= limits["paired_matching_minus_best_wrong_margin_mean_min"],
        "control_power": metrics["control_matching_accepts"] / metrics["control_outputs"]
        >= limits["control_matching_power_min"],
        "receipt_controls": (
            metrics["invalid_receipt_decision"] == limits["invalid_receipt_control"]
            and metrics["missing_receipt_decision"] == limits["missing_receipt_control"]
        ),
    }


def reconstruct(
    config: dict[str, Any], receipt_key: bytes, carrier_root: bytes, raw: Path,
) -> dict[str, Any]:
    public_rows = read_rows(raw / "PUBLIC_ROWS.jsonl")
    primitives = read_rows(raw / "EXECUTION_ROWS.jsonl")
    packages = read_rows(raw / "RECEIPT_PACKAGES.jsonl")
    outputs = read_rows(raw / "OUTPUT_ROWS.jsonl")
    public = {row["closure_id"]: row for row in public_rows}
    primitive_by_package = {row["package_id"]: row for row in primitives}
    primitive_by_side = {(row["closure_id"], row["role"]): row for row in primitives}
    protocol = read_json(RECEIPT_PROTOCOL_PATH)
    authority = read_json(AUTHORITY_PATH)

    receipt_results = []
    for package in packages:
        primitive = primitive_by_package[package["package_id"]]
        receipt_results.append(
            REC_REF.evaluate_bundle(
                receipt_key, package["receipts"],
                expected_binding(primitive, config), protocol,
            )
        )
    state_keys = [state_authority(carrier_root, value, authority) for value in range(16)]
    surface = A0.load_checkpoint(ROOT / config["lineage"]["A0_checkpoint_path"])[0]
    summaries = []
    generation_matches = 0
    control_scores = 0
    for output in outputs:
        decisions = [
            {"category": category, **A0_REF.detect(
                output["literal_output"], key, config["channel"]["threshold"]
            )}
            for category, key in enumerate(state_keys)
        ]
        matching = decisions[output["source_z2_category"]]
        wrong = [
            item for item in decisions
            if item["category"] != output["source_z2_category"]
        ]
        own = None
        if output["condition"] in ("PROMPT", "ANSWER", "GENERIC"):
            own = A0_REF.detect(
                output["literal_output"],
                authority_for(output, carrier_root, authority, public),
                config["channel"]["threshold"],
            )
            control_scores += 1
        parsed = A0_REF.parse(output["literal_output"])
        semantic_ok = (
            parsed["valid"]
            and parsed["values"]
            == {field: output["semantic_payload"][field] for field in A0_REF.FIELDS}
        )
        regenerated = A0.generate(
            surface,
            output["semantic_payload"],
            stream_seed=paired_nonce(output["closure_id"], output["draw"]),
            target_key=authority_for(output, carrier_root, authority, public),
            carrier_enabled=output["condition"] != "OFF",
            bias=config["channel"]["bias"],
            retain_operations=False,
        )
        generation_match = regenerated["literal_output"] == output["literal_output"]
        generation_matches += int(generation_match)
        summaries.append({
            "closure_id": output["closure_id"],
            "side": output["side"],
            "condition": output["condition"],
            "draw": output["draw"],
            "matching_decision": matching["decision"],
            "matching_score": matching["score"],
            "max_wrong_score": max(item["score"] for item in wrong),
            "wrong_accept_count": sum(item["decision"] == "ACCEPT" for item in wrong),
            "all_state_accept_count": sum(
                item["decision"] == "ACCEPT" for item in decisions
            ),
            "own_family_decision": None if own is None else own["decision"],
            "semantic_valid": semantic_ok,
        })
    output_index = {
        (row["closure_id"], row["side"], row["condition"], row["draw"]): row
        for row in outputs
    }
    payload_equal = nonce_equal = paired_total = 0
    baseline_equal = baseline_total = 0
    for closure_id in public:
        for condition, draws in (
            ("STATE", range(4)), ("OFF", range(4)), ("PROMPT", range(1)),
            ("ANSWER", range(1)), ("GENERIC", range(1)),
        ):
            for draw in draws:
                paired_total += 1
                left = output_index[(closure_id, "NATURAL", condition, draw)]
                right = output_index[(closure_id, "ALIAS_Z2", condition, draw)]
                payload_equal += int(
                    left["semantic_payload_sha256"] == right["semantic_payload_sha256"]
                )
                nonce_equal += int(
                    left["stream_seed_sha256"] == right["stream_seed_sha256"]
                )
                if condition != "STATE":
                    baseline_total += 1
                    baseline_equal += int(left["literal_output"] == right["literal_output"])
    baseline_authority_equal = baseline_authority_total = 0
    for row in public_rows:
        for family in ("PROMPT", "ANSWER", "GENERIC"):
            baseline_authority_total += 1
            left = baseline_authority(
                carrier_root, family, row["prompt_text"], row["natural"]["answer"]
            )
            right = baseline_authority(
                carrier_root, family, row["prompt_text"], row["natural"]["answer"]
            )
            baseline_authority_equal += int(left == right)
    categories_to_exact: dict[int, set[str]] = defaultdict(set)
    category_authorities: dict[int, set[bytes]] = defaultdict(set)
    for package in packages:
        category = package["receipts"]["abstract_z2"]["preimage"]["category"]
        categories_to_exact[category].add(package["receipts"]["exact_z2"]["receipt_id"])
        category_authorities[category].add(
            state_authority(carrier_root, category, authority)
        )
    state_rows = [row for row in summaries if row["condition"] == "STATE"]
    off_rows = [row for row in summaries if row["condition"] == "OFF"]
    controls = [
        row for row in summaries if row["condition"] in ("PROMPT", "ANSWER", "GENERIC")
    ]
    side_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        side_groups[(row["closure_id"], row["side"])].append(row)
    side_success = {
        key: (
            sum(item["matching_decision"] == "ACCEPT" for item in values) >= 3
            and all(item["wrong_accept_count"] == 0 for item in values)
        )
        for key, values in side_groups.items()
    }
    closure_success = [
        side_success[(closure_id, "NATURAL")]
        and side_success[(closure_id, "ALIAS_Z2")]
        for closure_id in public
    ]
    invalid = copy.deepcopy(packages[0])
    invalid["receipts"]["abstract_z2"]["tag"] = "0" * 64
    invalid_decisions = REC_REF.evaluate_bundle(
        receipt_key, invalid["receipts"],
        expected_binding(primitive_by_package[invalid["package_id"]], config), protocol,
    )
    missing = copy.deepcopy(packages[0]["receipts"])
    missing["abstract_z2"] = None
    missing_decisions = REC_REF.evaluate_bundle(
        receipt_key, missing,
        expected_binding(primitive_by_package[packages[0]["package_id"]], config),
        protocol,
    )
    return {
        "closures": len(public_rows),
        "execution_packages": len(packages),
        "receipt_envelopes": len(packages) * 5,
        "generated_outputs": len(outputs),
        "state_authority_scores": len(outputs) * 16,
        "own_family_control_scores": control_scores,
        "detector_decisions": len(outputs) * 16 + control_scores,
        "eligible_positions": sum(row["eligible_positions"] for row in outputs),
        "receipt_and_chain_accepts": sum(
            all(value == "ACCEPT" for value in row.values())
            for row in receipt_results
        ),
        "all16_authorities_distinct": len(set(state_keys)) == 16,
        "same_state_categories_with_multiple_exact_executions": sum(
            len(values) >= 2 for values in categories_to_exact.values()
        ),
        "same_state_authority_equal_categories": sum(
            len(values) == 1 for values in category_authorities.values()
        ),
        "payload_pair_equal": payload_equal,
        "payload_pair_total": paired_total,
        "paired_nonce_equal": nonce_equal,
        "paired_nonce_total": paired_total,
        "baseline_authority_equal": baseline_authority_equal,
        "baseline_authority_total": baseline_authority_total,
        "baseline_output_equal": baseline_equal,
        "baseline_output_total": baseline_total,
        "state_authority_switches": sum(
            primitive_by_side[(closure, "NATURAL")]["consumed_z2"]
            != primitive_by_side[(closure, "ALIAS_Z2")]["consumed_z2"]
            for closure in public
        ),
        "semantic_successes": sum(row["semantic_valid"] for row in summaries),
        "generation_reconstruction_matches": generation_matches,
        "matching_state_accepts": sum(
            row["matching_decision"] == "ACCEPT" for row in state_rows
        ),
        "state_outputs": len(state_rows),
        "state_outputs_with_any_wrong_accept": sum(
            row["wrong_accept_count"] > 0 for row in state_rows
        ),
        "carrier_off_outputs_with_any_state_accept": sum(
            row["all_state_accept_count"] > 0 for row in off_rows
        ),
        "carrier_off_outputs": len(off_rows),
        "side_successes": sum(side_success.values()),
        "sides": len(side_success),
        "closure_pair_switch_successes": sum(closure_success),
        "paired_margin_mean": sum(
            row["matching_score"] - row["max_wrong_score"] for row in state_rows
        ) / len(state_rows),
        "control_matching_accepts": sum(
            row["own_family_decision"] == "ACCEPT" for row in controls
        ),
        "control_outputs": len(controls),
        "invalid_receipt_decision": invalid_decisions["abstract_z2_state"],
        "missing_receipt_decision": missing_decisions["abstract_z2_state"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--raw-commit", required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--reveal-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        config, receipt, carrier = validate(
            args.source_commit, args.raw, args.raw_commit,
            args.reveal, args.reveal_commit,
        )
        metrics = reconstruct(config, receipt, carrier, args.raw)
        gate_rows = gates(config, metrics)
        result = {
            "schema": "exp33a.csw_am_scb_m1_car_d0.reference_result.v1",
            "metric_core": metrics,
            "metric_core_digest": digest(metrics),
            "gates": gate_rows,
            "gate_digest": digest(gate_rows),
            "passed": all(gate_rows.values()),
        }
        write_json(args.output, result)
    except BaseException as error:
        print(json.dumps({"error": type(error).__name__, "message": str(error)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
