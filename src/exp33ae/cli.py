"""Command-line entry points for explicit DEV_ONLY phases."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("task-audit")
    train = sub.add_parser("train")
    train.add_argument("--profile", choices=("smoke", "full"), default="full")
    sub.add_parser("search")
    sub.add_parser("register-calls")
    sub.add_parser("evaluate")
    sub.add_parser("polarity")
    sub.add_parser("report")
    sub.add_parser("index")
    test = sub.add_parser("test")
    test.add_argument("--output")
    audit = sub.add_parser("audit")
    audit.add_argument("--root")
    audit.add_argument("--output")
    pc_build = sub.add_parser("preconfirm-build")
    pc_build.add_argument("--output")
    pc_polarity = sub.add_parser("preconfirm-polarity")
    pc_polarity.add_argument("--fixture")
    pc_polarity.add_argument("--output", required=True)
    pc_polarity.add_argument("--shard-index", type=int, default=0)
    pc_polarity.add_argument("--shard-count", type=int, default=1)
    pc_finalize = sub.add_parser("preconfirm-finalize")
    pc_finalize.add_argument("--output")
    pc_audit = sub.add_parser("preconfirm-audit")
    pc_audit.add_argument("--root")
    pc_audit.add_argument("--output", required=True)
    pc_audit.add_argument("--pretag", action="store_true")
    pc_audit.add_argument("--self-contained-clone", action="store_true")
    remediation_run = sub.add_parser("remediation-run")
    remediation_run.add_argument("--output", required=True)
    remediation_run.add_argument("--master-hex", required=True)
    remediation_run.add_argument("--stage-id", required=True)
    remediation_run.add_argument("--source-commit", required=True)
    remediation_run.add_argument("--resume", action="store_true")
    remediation_run.add_argument("--stop-after-calls", type=int)
    remediation_polarity = sub.add_parser("remediation-polarity")
    remediation_polarity.add_argument("--package", required=True)
    remediation_polarity.add_argument("--output")
    remediation_audit = sub.add_parser("remediation-audit")
    remediation_audit.add_argument("--package", required=True)
    remediation_audit.add_argument("--output")
    remediation_audit.add_argument("--pretag", action="store_true")
    reliability_prepare = sub.add_parser("reliability-prepare")
    reliability_prepare.add_argument("--output", required=True)
    reliability_prepare.add_argument("--freeze-tag", default="dev-only-33a-e-reliability-study-freeze-v1")
    reliability_evaluate = sub.add_parser("reliability-evaluate")
    reliability_evaluate.add_argument("--output", required=True)
    reliability_audit = sub.add_parser("reliability-audit")
    reliability_audit.add_argument("--output", required=True)
    reliability_audit.add_argument("--report")
    reliability_audit.add_argument("--pretag", action="store_true")
    cleanup_preflight = sub.add_parser("reliability-cleanup-preflight")
    cleanup_projection = sub.add_parser("reliability-cleanup-project")
    cleanup_projection.add_argument("--output")
    cleanup_audit = sub.add_parser("reliability-cleanup-audit")
    cleanup_audit.add_argument("--pretag", action="store_true")
    sub.add_parser("confirmation-protocol-preflight")
    protocol_fixture = sub.add_parser("confirmation-protocol-dev-fixture")
    protocol_fixture.add_argument("--output")
    protocol_audit = sub.add_parser("confirmation-protocol-audit")
    protocol_audit.add_argument("--fast", action="store_true")
    confirmation_run = sub.add_parser("confirmation-run")
    confirmation_run.add_argument("--output", required=True)
    confirmation_run.add_argument("--external-seal", required=True)
    confirmation_run.add_argument("--external-authorization", required=True)
    confirmation_run.add_argument("--master-hex", required=True)
    confirmation_run.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "task-audit":
        from .task_audit import run_task_audit

        print(json.dumps(run_task_audit(materialize=True), indent=2, sort_keys=True))
    elif args.command == "train":
        from pathlib import Path
        from .artifacts import write_json
        from .training import train_population

        root = Path(__file__).resolve().parents[2]
        summaries = train_population(root, args.profile)
        output = {
            "schema": "exp33ae.training_population.v1",
            "namespace": "DEV_ONLY",
            "profile": args.profile,
            "claim_bearing": False,
            "models": summaries,
            "confirmation_artifacts_created": 0,
            "confirmation_artifacts_accessed": 0,
        }
        write_json(root / f"results/dev_only/training-{args.profile}.json", output)
        print(json.dumps(output, indent=2, sort_keys=True))
    elif args.command == "search":
        from pathlib import Path
        from .search import run_search

        result = run_search(Path(__file__).resolve().parents[2])
        print(json.dumps({"total_rows": result["total_rows"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
    elif args.command == "register-calls":
        from pathlib import Path
        from .registry import create_registry

        print(json.dumps(create_registry(Path(__file__).resolve().parents[2]), indent=2, sort_keys=True))
    elif args.command == "evaluate":
        from pathlib import Path
        from .evaluate import run_evaluation

        result = run_evaluation(Path(__file__).resolve().parents[2])
        print(json.dumps({
            "ordinary_aggregate": result["ordinary_aggregate"],
            "iit_primary_aggregate": result["iit_primary_aggregate"],
            "attempted_calls": result["attempted_calls"],
            "runtime_seconds": result["runtime_seconds"],
        }, indent=2, sort_keys=True))
    elif args.command == "polarity":
        from pathlib import Path
        from .polarity import run_polarity

        result = run_polarity(Path(__file__).resolve().parents[2])
        print(json.dumps({
            "scenario_count": result["scenario_count"],
            "passed_count": result["passed_count"],
        }, indent=2, sort_keys=True))
    elif args.command == "report":
        from pathlib import Path
        from .reporting import run_metric_reconstruction

        result = run_metric_reconstruction(Path(__file__).resolve().parents[2])
        print(json.dumps({"rows": len(result["rows"])}, indent=2, sort_keys=True))
    elif args.command == "index":
        from pathlib import Path
        from .indexing import build_artifact_index

        result = build_artifact_index(Path(__file__).resolve().parents[2])
        print(json.dumps({"file_count": result["file_count"]}, indent=2, sort_keys=True))
    elif args.command == "test":
        from pathlib import Path
        from .test_reporting import run_test_report

        root = Path(__file__).resolve().parents[2]
        output = Path(args.output).resolve() if args.output else root / "results/dev_only/test-report.json"
        result = run_test_report(root, output)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed_suite"]:
            raise SystemExit(1)
    elif args.command == "audit":
        from pathlib import Path
        from .audit import run_hostile_audit

        default_root = Path(__file__).resolve().parents[2]
        root = Path(args.root).resolve() if args.root else default_root
        output = Path(args.output).resolve() if args.output else root / "results/dev_only/hostile-readiness-audit.json"
        result = run_hostile_audit(root, output)
        print(json.dumps({
            "passed": result["passed"],
            "passed_checks": result["passed_checks"],
            "total_checks": result["total_checks"],
            "readiness_status": result["readiness"]["status"],
        }, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "preconfirm-build":
        from pathlib import Path
        from .preconfirmation.build import build_dev_only_fixture

        root = Path(__file__).resolve().parents[2]
        output = Path(args.output).resolve() if args.output else root / "artifacts/dev_only/preconfirmation-revision-v1"
        result = build_dev_only_fixture(root, output)
        print(json.dumps(result["report"], indent=2, sort_keys=True))
    elif args.command == "preconfirm-polarity":
        from pathlib import Path
        from .preconfirmation.polarity import run_polarity

        root = Path(__file__).resolve().parents[2]
        fixture = Path(args.fixture).resolve() if args.fixture else root / "artifacts/dev_only/preconfirmation-revision-v1"
        result = run_polarity(root, fixture, Path(args.output).resolve(), args.shard_index, args.shard_count)
        print(json.dumps({"scenario_count": result["scenario_count"], "passed_count": result["passed_count"]}, indent=2, sort_keys=True))
    elif args.command == "preconfirm-finalize":
        from pathlib import Path
        from .preconfirmation.release import finalize_layer_1

        root = Path(__file__).resolve().parents[2]
        output = Path(args.output).resolve() if args.output else root / "results/dev_only/preconfirmation-revision-v1"
        print(json.dumps(finalize_layer_1(root, output), indent=2, sort_keys=True))
    elif args.command == "preconfirm-audit":
        from pathlib import Path
        from .preconfirmation.audit import run_audit

        default_root = Path(__file__).resolve().parents[2]
        root = Path(args.root).resolve() if args.root else default_root
        result = run_audit(
            root, Path(args.output).resolve(), require_candidate_tag=not args.pretag,
            verify_live_predecessor=not args.self_contained_clone,
        )
        print(json.dumps({"passed": result["passed"], "passed_checks": result["passed_checks"], "total_checks": result["total_checks"], "status": result["status"]}, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "remediation-run":
        from pathlib import Path
        from .remediation.runner import run

        root = Path(__file__).resolve().parents[2]
        result = run(
            root, Path(args.output).resolve(), args.master_hex, args.stage_id,
            args.source_commit, resume=args.resume, stop_after_calls=args.stop_after_calls,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "remediation-polarity":
        from pathlib import Path
        from .remediation.polarity import run_polarity

        result = run_polarity(
            Path(args.package).resolve(),
            Path(args.output).resolve() if args.output else None,
        )
        print(json.dumps({"scenario_count": result["scenario_count"], "passed_count": result["passed_count"], "actual_rehearsal_status": result["actual_rehearsal_status"]}, indent=2, sort_keys=True))
        if result["passed_count"] != result["scenario_count"]:
            raise SystemExit(1)
    elif args.command == "remediation-audit":
        from pathlib import Path
        from .remediation.audit import run_audit

        root = Path(__file__).resolve().parents[2]
        result = run_audit(
            root, Path(args.package).resolve(),
            Path(args.output).resolve() if args.output else None,
            require_candidate_tag=not args.pretag,
        )
        print(json.dumps({"passed": result["passed"], "passed_checks": result["passed_checks"], "total_checks": result["total_checks"], "terminal_status": result["terminal_status"]}, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "reliability-prepare":
        from pathlib import Path
        from .reliability.runner import prepare

        root = Path(__file__).resolve().parents[2]
        result = prepare(root, Path(args.output).resolve(), args.freeze_tag)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "reliability-evaluate":
        from pathlib import Path
        from .reliability.runner import evaluate_all

        root = Path(__file__).resolve().parents[2]
        result = evaluate_all(root, Path(args.output).resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "reliability-audit":
        from pathlib import Path
        from .reliability.audit import run

        root = Path(__file__).resolve().parents[2]
        result = run(root, Path(args.output).resolve(), Path(args.report).resolve() if args.report else None,
                     require_candidate=not args.pretag)
        print(json.dumps({"passed": result["passed"], "passed_checks": result["passed_checks"],
                          "total_checks": result["total_checks"]}, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "reliability-cleanup-preflight":
        from .reliability.cleanup import dependency_preflight

        result = dependency_preflight()
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["contract_satisfied"]:
            raise SystemExit(1)
    elif args.command == "reliability-cleanup-project":
        from pathlib import Path
        from .reliability.cleanup import build_supplemental_manifest

        root = Path(__file__).resolve().parents[2]
        result = build_supplemental_manifest(root, Path(args.output).resolve() if args.output else None)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "reliability-cleanup-audit":
        from pathlib import Path
        from .reliability.cleanup import run_cleanup_audit

        root = Path(__file__).resolve().parents[2]
        result = run_cleanup_audit(root, require_candidate=not args.pretag)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "confirmation-protocol-preflight":
        from .confirmation_protocol.normative import candidate_counters, offline_dependency_preflight, protocol_hash
        result = {"protocol_sha256": protocol_hash(), "dependency": offline_dependency_preflight(),
                  "confirmation_counters": candidate_counters(), "confirmation_launch_attempted": False}
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["dependency"]["passed"]:
            raise SystemExit(1)
    elif args.command == "confirmation-protocol-dev-fixture":
        from pathlib import Path
        from .confirmation_protocol.fixture import build
        root = Path(__file__).resolve().parents[2]
        result = build(root, Path(args.output).resolve() if args.output else None)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "confirmation-protocol-audit":
        from pathlib import Path
        from .confirmation_protocol.audit import run
        root = Path(__file__).resolve().parents[2]
        result = run(root, full_fixture=not args.fast)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.command == "confirmation-run":
        from pathlib import Path
        from .remediation.strict import strict_load_json
        from .confirmation_protocol.runner import run_confirmation
        root = Path(__file__).resolve().parents[2]
        result = run_confirmation(root, Path(args.output).resolve(), strict_load_json(Path(args.external_seal).resolve()),
                                  strict_load_json(Path(args.external_authorization).resolve()), args.master_hex, resume=args.resume)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
