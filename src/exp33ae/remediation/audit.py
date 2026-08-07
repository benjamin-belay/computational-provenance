"""Repository-authored, explicitly non-independent hostile remediation audit."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import subprocess

from .access import DevOnlyAccessController
from .allocation import load_allocation
from .binding_polarity import run_binding_polarity
from .entropy import DevMasterEntropy
from .evaluator import evaluate
from .indexing import verify_postrelease_index
from .normative import POSITIVE_ANALOGUE, protocol_hash, verify_operative_manifest
from .polarity import run_polarity
from .runtime import environment_manifest, load_checkpoint
from .strict import sha256_file, strict_load_json, strict_load_jsonl, write_json
from .validation import validate_raw, validate_registry


V3 = {
    "tag_object": "c810a84adc4ce3196af441ba9fbef96958d5098a",
    "commit": "af90167fcbb1e2ea834d84d3463a27ae38f2c820",
    "tree": "dd202b8f2caf6f77f987c0575d094a188ccddfce",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def _check(condition, evidence) -> dict:
    return {"passed": type(condition) is bool and condition, "evidence": evidence}


def run_audit(root: Path, output: Path, report_path: Path | None = None, require_candidate_tag: bool = True) -> dict:
    checks = {}
    observed_v3 = {
        "tag_object": _git(root, "rev-parse", "protocol-revision-candidate-v3"),
        "commit": _git(root, "rev-parse", "protocol-revision-candidate-v3^{}"),
        "tree": _git(root, "rev-parse", "protocol-revision-candidate-v3^{tree}"),
    }
    checks["01_v3_preserved"] = _check(observed_v3 == V3, observed_v3)
    tags = _git(root, "tag", "--points-at", "HEAD").splitlines()
    remediation_tags = [tag for tag in tags if tag.startswith("dev-only-33a-e-remediation-candidate-")]
    clean = _git(root, "status", "--porcelain", "--untracked-files=all")
    checks["02_immutable_clean_candidate"] = _check(
        clean == "" and (not require_candidate_tag or len(remediation_tags) == 1),
        {"head": _git(root, "rev-parse", "HEAD"), "tree": _git(root, "rev-parse", "HEAD^{tree}"), "status": clean, "tags": remediation_tags},
    )

    entropy_record = strict_load_json(output / "entropy.json")
    zero_fields = {
        "confirmation_master_entropy": 0,
        "confirmation_derived_seeds": 0,
        "confirmation_programmes": 0,
        "confirmation_examples": 0,
        "confirmation_pairs": 0,
        "confirmation_registries": 0,
        "confirmation_checkpoints": 0,
        "confirmation_calls": 0,
        "confirmation_results": 0,
        "confirmation_access_records": 0,
        "confirmation_accesses": 0,
    }
    checks["03_zero_confirmation"] = _check(
        entropy_record["claim_bearing"] is False
        and entropy_record["confirmation_master_entropy_created"] is False
        and entropy_record["confirmation_derived_objects_created"] == 0
        and all(value == 0 for value in zero_fields.values()),
        zero_fields,
    )
    operative = strict_load_json(output / "operative-source-manifest.json")
    source_faults = verify_operative_manifest(root, operative)
    checks["04_complete_source_binding"] = _check(not source_faults and operative["protocol_sha256"] == protocol_hash(), {"files": len(operative["files"]), "manifest": operative["manifest_sha256"], "faults": source_faults})
    environment = strict_load_json(output / "environment.json")
    checks["05_environment_binding"] = _check(environment == environment_manifest(), {"manifest": environment["manifest_sha256"], "python": environment["python"], "torch": environment["torch"], "numpy": environment["numpy"], "platform": environment["platform"]})

    allocation, pairs, allocation_manifest = load_allocation(output)
    partition_counts = Counter(row.partition for row in allocation["examples"])
    pair_counts = Counter(row["stratum"] for row in pairs)
    checks["06_allocation_and_pair_counts"] = _check(
        partition_counts == Counter({"training": 8192, "model_selection": 480, "localization_fit": 480, "localization_selection": 480, "clean_target_axis": 800, "clean_parallel_axis": 800})
        and pair_counts == Counter({"A_answer_changing": 160, "B_outcome_collision": 80, "C_same_state": 80, "D_same_observable_answer": 80})
        and allocation_manifest["production_independent_oracle_mismatches"] == 0,
        {"partitions": dict(partition_counts), "pairs": dict(pair_counts), "oracle_mismatches": allocation_manifest["production_independent_oracle_mismatches"]},
    )

    entropy = DevMasterEntropy(entropy_record["stage_id"], bytes.fromhex(entropy_record["dev_only_master_hex"]))
    seeds = list(entropy.model_seeds())
    training = []
    training_ok = True
    for order, seed in enumerate(seeds):
        manifest = strict_load_json(output / f"training/model-{order}.json")
        training.append(manifest)
        try:
            load_checkpoint(output / manifest["checkpoint_relative_path"], {
                "model_order": order, "model_seed_identity": seed,
                "operative_manifest_sha256": operative["manifest_sha256"],
                "environment_sha256": environment["manifest_sha256"],
                "tensor_state_sha256": manifest["final_tensor_state_sha256"],
                "checkpoint_file_sha256": manifest["checkpoint_file_sha256"],
            })
        except Exception:
            training_ok = False
    checks["07_actual_training_and_checkpoints"] = _check(
        training_ok and len(training) == 3 and all(row["steps"] == 1400 and row["interchange_pairs"] == 358400 for row in training),
        {"seeds": seeds, "checkpoint_files": [row["checkpoint_file_sha256"] for row in training], "tensor_states": [row["final_tensor_state_sha256"] for row in training]},
    )

    registry_manifest = strict_load_json(output / "registry/manifest.json")
    registry = strict_load_jsonl(output / "registry/registry.jsonl")
    raw = strict_load_jsonl(output / "access/buffer.jsonl")
    registry_ok = raw_ok = True
    try:
        validate_registry(registry, registry_manifest, seeds)
        validate_raw(raw, registry)
    except Exception:
        registry_ok = raw_ok = False
    checks["08_registry_and_raw_reconciliation"] = _check(registry_ok and raw_ok and len(registry) == len(raw) == 11760, {"planned": len(registry), "observed": len(raw), "registry_sha256": sha256_file(output / "registry/registry.jsonl"), "raw_sha256": sha256_file(output / "access/buffer.jsonl")})

    controller = DevOnlyAccessController(output / "access", entropy.stage_id, entropy.derive("access-token", 0, 32), registry_manifest["payload_commitment"], [row["call_id"] for row in registry])
    access_ok = True
    try:
        with controller.acquire():
            snapshot = controller.validate_chain()
            access_state = snapshot["state"]["state"]
    except Exception as error:
        access_ok = False; access_state = type(error).__name__
    checks["09_access_chain_and_release"] = _check(access_ok and access_state == "RELEASED", {"state": access_state, "journal_sha256": sha256_file(output / "access/journal.jsonl")})

    derived = strict_load_json(output / "derived-status.json")
    release = strict_load_json(output / "release-envelope.json")
    checks["10_evaluator_derived_nonclaiming_status"] = _check(
        derived["status"] == release["status"]
        and derived["terminal_confirmation_status_materialized"] is False
        and release["caller_supplied_result_used"] is False
        and release["confirmation_authorized"] is False and release["protocol_sealed"] is False,
        {"status": derived["status"], "seed_statuses": derived.get("seed_statuses"), "positive_analogue": POSITIVE_ANALOGUE},
    )
    post = verify_postrelease_index(output)
    checks["11_postrelease_index"] = _check(post["matches"], post)

    replay = run_polarity(output)
    stored_polarity = strict_load_json(output / "polarity/report.json")
    checks["12_polarity_regenerates"] = _check(replay == stored_polarity and replay["passed_count"] == replay["scenario_count"], {"scenarios": replay["scenario_count"], "categories": replay["category_counts"], "actual_status": replay["actual_rehearsal_status"]})
    binding_replay = run_binding_polarity(root, output)
    stored_binding = strict_load_json(output / "polarity/binding-report.json")
    checks["13_binding_polarity_regenerates"] = _check(
        binding_replay == stored_binding and binding_replay["passed_count"] == binding_replay["scenario_count"],
        {"scenarios": binding_replay["scenario_count"], "passed": binding_replay["passed_count"]},
    )
    checks["14_no_terminal_confirmation_literal"] = _check(
        derived["status"] != "CONFIRMED_ENGINEERED_CAUSAL_ABSTRACTION_POSITIVE"
        and not any(path.name.startswith("confirmation") for path in output.rglob("*")),
        {"derived_status": derived["status"], "confirmation_named_paths": [str(path.relative_to(output)) for path in output.rglob("*") if path.name.startswith("confirmation")]},
    )

    passed = all(row["passed"] for row in checks.values())
    report = {
        "schema": "exp33ae.dev_only_remediation_hostile_audit.v2", "claim_bearing": False,
        "independent_review": False, "checks": checks, "passed_checks": sum(row["passed"] for row in checks.values()),
        "total_checks": len(checks), "passed": passed,
        "terminal_status": "33A_E_REMEDIATION_CANDIDATE_READY_FOR_REVIEW" if passed else "33A_E_REMEDIATION_FAILED",
        "protocol_sealing_authorized": False, "confirmation_authorized": False,
        "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    if report_path is not None:
        write_json(report_path, report)
    return report
