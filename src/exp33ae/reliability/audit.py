"""Repository-authored, explicitly non-independent reliability candidate audit."""

from __future__ import annotations

from pathlib import Path

from ..remediation.strict import sha256_file, strict_load_json, strict_load_jsonl, write_json
from .evaluator import load_training, validate_registry
from .entropy import StudyEntropy
from .indexing import verify_indexes
from .normative import BASELINE_COMMIT, CANDIDATE_TAG, DEV_NAMESPACE, FREEZE_TAG, git, load_protocol


def run(root: Path, output: Path, report_path: Path | None = None, require_candidate: bool = True) -> dict:
    checks = []
    def check(name, passed, evidence): checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
    baseline_tree = git(root, "rev-parse", f"{BASELINE_COMMIT}^{{tree}}")
    check("baseline_preserved", git(root, "rev-parse", "dev-only-33a-e-remediation-candidate-v1^{commit}") == BASELINE_COMMIT,
          {"commit": BASELINE_COMMIT, "tree": baseline_tree})
    freeze_commit = git(root, "rev-parse", f"{FREEZE_TAG}^{{commit}}"); check("preresult_freeze_exists", bool(freeze_commit), freeze_commit)
    entropy_record = strict_load_json(output / "entropy.json"); entropy = StudyEntropy(freeze_commit)
    check("deterministic_master", entropy_record["dev_only_master_hex"] == entropy.master.hex(), entropy.master.hex())
    protocol = load_protocol(); check("exact_recipes", [(r["d_model"],r["d_mlp"],r["steps"]) for r in protocol["recipes"]] == [(64,128,1400),(64,128,2800),(96,192,1400),(96,192,2800)], protocol["recipes"])
    training = load_training(output); check("exact_models", len(training) == 20 and [(r["recipe_order"],r["seed_order"]) for r in training] == [(i,j) for i in range(4) for j in range(5)], len(training))
    registry_manifest = strict_load_json(output / "registry" / "manifest.json"); registry = strict_load_jsonl(output / "registry" / "registry.jsonl")
    try: validate_registry(registry, registry_manifest, training, entropy); registry_ok = True
    except Exception as error: registry_ok = False
    check("exact_registry", registry_ok and len(registry) == 78400, {"rows": len(registry), "sha256": sha256_file(output / "registry" / "registry.jsonl")})
    status = strict_load_json(output / "derived-status.json"); check("status_mechanical", status["status"] in {"RELIABLE_TRAINING_RECIPE_IDENTIFIED_DEV_ONLY","NO_RELIABLE_TRAINING_RECIPE_FOUND"} and len(status["models"]) == 20, status["status"])
    sci, runtime = verify_indexes(output); check("scientific_runtime_indexes", sci and runtime, {"scientific":sci,"runtime":runtime})
    check("zero_confirmation", status["confirmation_objects_created"] == status["confirmation_objects_accessed"] == 0 and not any("confirmation" in p.name.lower() for p in output.rglob("*") if p.is_file()), {"created":0,"accessed":0})
    historical = (root / "docs/HISTORICAL_NON_OPERATIVE_NOTICE.md").read_text(); check("historical_cleanup", "48/160 = 0.30" in historical and "DAS and probes cannot" in historical, "notice")
    check("dependency_boundary", "not independent dependency reconstruction" in strict_load_json(output / "dependency-manifest.json")["boundary"], "locked local environment")
    if require_candidate:
        check("candidate_tag", git(root, "rev-parse", f"{CANDIDATE_TAG}^{{commit}}") == git(root, "rev-parse", "HEAD^{commit}"), CANDIDATE_TAG)
    report = {"schema": "exp33ae.dev_only_reliability_non_independent_audit.v1", "namespace": DEV_NAMESPACE,
              "claim_bearing": False, "independent": False, "passed": all(c["passed"] for c in checks),
              "passed_checks": sum(c["passed"] for c in checks), "total_checks": len(checks), "checks": checks,
              "confirmation_objects_created": 0, "confirmation_objects_accessed": 0}
    if report_path: write_json(report_path, report)
    return report
