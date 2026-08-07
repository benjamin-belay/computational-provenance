"""Replayable real-upstream DEV_ONLY polarity scenarios for every future branch."""

from __future__ import annotations

import copy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import torch

from ..artifacts import canonical_json, sha256_file, write_json
from ..model import ModelConfig, TinyProgrammeTransformer
from .access import AccessViolation, DevOnlyAccessSession
from .allocator import ProtocolExample
from .entropy import dev_golden_entropy
from .evaluator import FutureStatus, evaluate
from .normative import DEV_NAMESPACE, protocol_hash


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_fixture_package(root: Path, fixture: Path) -> dict:
    entropy = dev_golden_entropy()
    examples = []
    for row in _read_jsonl(fixture / "example-allocation.jsonl"):
        row["operation_ids"] = tuple(row["operation_ids"])
        examples.append(ProtocolExample(**row))
    allocation_audit = json.loads((fixture / "allocation-audit.json").read_text())
    allocation = {
        "schema": "exp33ae.dev_only_allocation.v1", "namespace": DEV_NAMESPACE, "stage_id": entropy.stage_id,
        "claim_bearing": False, "programmes": _read_jsonl(fixture / "programme-allocation.jsonl"),
        "examples": examples,
    }
    registry_manifest = json.loads((fixture / "registry-manifest.json").read_text())
    registry = {
        "schema": registry_manifest["schema"], "namespace": DEV_NAMESPACE, "stage_id": entropy.stage_id,
        "claim_bearing": False, "model_seed_identities": registry_manifest["model_seed_identities"],
        "payload_commitment": registry_manifest["payload_commitment"], "rows": _read_jsonl(fixture / "registry.jsonl"),
    }
    exclusions = json.loads((fixture / "development-exclusions.json").read_text())
    exclusions = {key: value for key, value in exclusions.items() if key not in {"schema", "namespace"}}
    source = json.loads((fixture / "source-schema-manifest.json").read_text())["manifest_sha256"]
    readouts = json.loads((fixture / "readout-manifest.json").read_text())["readouts"]
    access = json.loads((fixture / "access-happy-report.json").read_text())
    return {
        "schema": "exp33ae.dev_only_evaluation_package.v1", "namespace": DEV_NAMESPACE,
        "stage_id": entropy.stage_id, "claim_bearing": False, "protocol_sha256": protocol_hash(),
        "source_manifest_expected": source, "source_manifest_observed": source,
        "payload_index_expected": registry["payload_commitment"], "payload_index_observed": registry["payload_commitment"],
        "requested_status": None, "access_report": access, "allocation": allocation,
        "registry": registry, "raw_rows": _read_jsonl(fixture / "raw-results.jsonl"),
        "expected_model_seed_identities": list(entropy.ordered_model_seeds()),
        "development_exclusions": exclusions, "readout_manifest": readouts,
        "shortcut_baselines": allocation_audit["shortcut_baselines"],
    }


def _recommit(package: dict) -> None:
    payload = []
    for row in package["registry"]["rows"]:
        copy_row = dict(row); copy_row.pop("registry_payload_commitment", None); payload.append(copy_row)
    commitment = sha256(canonical_json(payload).encode()).hexdigest()
    package["registry"]["payload_commitment"] = commitment
    package["payload_index_expected"] = commitment
    package["payload_index_observed"] = commitment
    for row in package["registry"]["rows"]:
        row["registry_payload_commitment"] = commitment


def _raw_by_call(package: dict) -> dict[str, dict]:
    return {row["call_id"]: row for row in package["raw_rows"]}


def _force_correct(package: dict, selector, raw_field: str, expected_field: str, target: int) -> None:
    plans = sorted((row for row in package["registry"]["rows"] if selector(row)), key=lambda row: row["call_id"])
    raw = _raw_by_call(package)
    if not 0 <= target <= len(plans):
        raise ValueError("invalid target correct count")
    for index, plan in enumerate(plans):
        expected = plan[expected_field]
        raw[plan["call_id"]][raw_field] = expected if index < target else (expected + 1) % (2 if "z2" in expected_field or "z3" in expected_field or "n3" in expected_field else 4)


def _run_actual_incompetent_checkpoint(package: dict, fixture: Path) -> None:
    manifest = json.loads((fixture / "incompetent-checkpoint-manifest.json").read_text())
    payload = torch.load(fixture / manifest["path"], map_location="cpu", weights_only=False)
    model = TinyProgrammeTransformer(ModelConfig(**payload["model_config"]))
    model.load_state_dict(payload["state_dict"]); model.eval()
    seed = package["expected_model_seed_identities"][0]
    examples = {row.example_id: row for row in package["allocation"]["examples"]}
    clean = [row for row in package["registry"]["rows"] if row["model_seed_identity"] == seed and row["method"] == "clean_task_decode"]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(clean), 256):
            batch = clean[start:start+256]
            rows = [examples[plan["base_id"]] for plan in batch]
            tokens = torch.tensor([row.tokens for row in rows], dtype=torch.long)
            tables = torch.tensor([row.operation_tables for row in rows], dtype=torch.float32)
            predictions.extend(model(tokens, tables).argmax(-1).tolist())
    raw = _raw_by_call(package)
    checkpoint_hash = manifest["file_sha256"]
    for plan, prediction in zip(clean, predictions):
        plan["checkpoint_sha256"] = checkpoint_hash
        raw[plan["call_id"]]["checkpoint_sha256"] = checkpoint_hash
        raw[plan["call_id"]]["intervened_prediction"] = prediction
    # Bind every seed-0 call to the same actual DEV_ONLY checkpoint identity.
    for plan in package["registry"]["rows"]:
        if plan["model_seed_identity"] == seed:
            plan["checkpoint_sha256"] = checkpoint_hash
            raw[plan["call_id"]]["checkpoint_sha256"] = checkpoint_hash
    _recommit(package)


def run_polarity(root: Path, fixture: Path, output: Path, shard_index: int = 0, shard_count: int = 1) -> dict:
    if not (type(shard_index) is int and type(shard_count) is int and 0 <= shard_index < shard_count):
        raise ValueError("invalid deterministic polarity shard")
    base = load_fixture_package(root, fixture)
    scenarios = []
    scenario_ordinal = 0

    def scenario(name: str, expected: str, mutate, upstream: str, exact_target: dict | None = None):
        nonlocal scenario_ordinal
        ordinal = scenario_ordinal
        scenario_ordinal += 1
        if ordinal % shard_count != shard_index:
            return
        package = copy.deepcopy(base)
        mutate(package)
        decision = evaluate(package, root)
        row = {
            "ordinal": ordinal, "scenario": name, "upstream_object": upstream, "expected_simulated_status": expected,
            "actual_simulated_status": decision["simulated_status"], "passed": decision["simulated_status"] == expected,
            "faults": decision["faults"], "exact_integer_target": exact_target,
            "direct_metric_dictionary_or_terminal_status_edit": False,
        }
        scenarios.append(row)

    positive = FutureStatus.POSITIVE.value; invalid = FutureStatus.INVALID.value
    competence = FutureStatus.COMPETENCE_FAILED.value; no_eligible = FutureStatus.NO_ELIGIBLE.value
    decode_only = FutureStatus.DECODE_ONLY.value; negative = FutureStatus.NEGATIVE.value
    scenario("healthy_positive_branch", positive, lambda p: None, "complete DEV_ONLY fixture package")
    scenario("incompetent_checkpoint", competence, lambda p: _run_actual_incompetent_checkpoint(p, fixture), "actual zeroed DEV_ONLY transformer checkpoint")
    scenario("checkpoint_state_drift", invalid, lambda p: p["raw_rows"][0].__setitem__("checkpoint_sha256", "0"*64), "raw checkpoint identity")

    def allocation_overlap(p):
        p["allocation"]["programmes"][0]["programme_id"] = p["development_exclusions"]["programme_ids"][0]
    scenario("allocation_overlap", invalid, allocation_overlap, "programme allocation")
    scenario("allocation_imbalance", invalid, lambda p: p["allocation"]["examples"].__setitem__(0, replace(p["allocation"]["examples"][0], z2=1-p["allocation"]["examples"][0].z2)), "example allocation")
    scenario("wrong_programme_signature", invalid, lambda p: p["allocation"]["programmes"][0].__setitem__("causal_signature", "0"*64), "programme allocation")

    def registry_mutation(field, value):
        def mutate(p):
            row = next(row for row in p["registry"]["rows"] if row["method"] == "declared_iit_alignment")
            row[field] = value(row) if callable(value) else value; _recommit(p)
        return mutate
    scenario("source_base_ineligibility", invalid, registry_mutation("source_id", lambda r: r["base_id"]), "source/base call plan")
    scenario("oracle_mismatch", invalid, registry_mutation("expected_answer", lambda r: (r["expected_answer"]+1)%4), "independent-oracle expectation")
    scenario("missing_call", invalid, lambda p: p["raw_rows"].pop(), "raw call ledger")
    scenario("duplicate_call", invalid, lambda p: p["raw_rows"].append(copy.deepcopy(p["raw_rows"][0])), "raw call ledger")
    def unplanned(p):
        row=copy.deepcopy(p["raw_rows"][0]); row["call_id"]="devonly-unplanned"; p["raw_rows"].append(row)
    scenario("unplanned_call", invalid, unplanned, "raw call ledger")
    scenario("failed_call", invalid, lambda p: p["raw_rows"][0].__setitem__("completed", False), "raw call result")
    scenario("retried_call", invalid, lambda p: p["raw_rows"][0].__setitem__("attempt_number", 2), "raw call result")
    scenario("dropped_call", invalid, lambda p: p["raw_rows"][0].__setitem__("dropped", True), "raw call result")
    def incomplete_control(p):
        call=next(row["call_id"] for row in p["registry"]["rows"] if row["method"]=="random_direction"); p["registry"]["rows"]=[r for r in p["registry"]["rows"] if r["call_id"]!=call]; p["raw_rows"]=[r for r in p["raw_rows"] if r["call_id"]!=call]; _recommit(p)
    scenario("incomplete_control_family", invalid, incomplete_control, "registry and raw calls")
    def empty_control(p):
        calls={r["call_id"] for r in p["registry"]["rows"] if r["method"]=="orthogonal_direction"}; p["registry"]["rows"]=[r for r in p["registry"]["rows"] if r["call_id"] not in calls]; p["raw_rows"]=[r for r in p["raw_rows"] if r["call_id"] not in calls]; _recommit(p)
    scenario("empty_control_family", invalid, empty_control, "registry and raw calls")
    def imbalanced_pair(p):
        row=next(row for row in p["registry"]["rows"] if row["method"]=="declared_iit_alignment" and row["pair_stratum"]=="B_outcome_collision"); row["pair_stratum"]="A_answer_changing"; _recommit(p)
    scenario("imbalanced_pair_strata", invalid, imbalanced_pair, "primary pair registry")
    def imbalanced_control(p):
        row=next(row for row in p["registry"]["rows"] if row["method"]=="random_direction"); row["control_id"]="devonly-random_direction-99"; _recommit(p)
    scenario("imbalanced_control_strata", invalid, imbalanced_control, "control registry")
    norm=lambda p: next(row for row in p["raw_rows"] if row["method"]=="random_direction").__setitem__("applied_norm",1.1)
    scenario("norm_mismatch", invalid, norm, "applied activation norm")
    for field, value in (("method","wrong_method"),("basis_sha256","0"*64),("layer",3),("position",16),("rank",1),("strength",0.75),("checkpoint_sha256","1"*64),("model_seed_identity",-1),("programme_id","wrong"),("pair_id","wrong"),("call_id","wrong"),("nonce","wrong")):
        scenario("wrong_"+field, invalid, lambda p,f=field,v=value: p["raw_rows"][0].__setitem__(f,v), "raw call "+field)

    access_root = output.parent / f"polarity-access-fixtures-shard-{shard_index}"
    second = DevOnlyAccessSession(access_root / "second-access", base["stage_id"], [row["call_id"] for row in base["registry"]["rows"]])
    second.authorize_once()
    try: second.authorize_once()
    except AccessViolation: pass
    scenario("real_second_access", invalid, lambda p: p.__setitem__("access_report", second.report()), "append-only access journal")
    partial = DevOnlyAccessSession(access_root / "partial-inspection", base["stage_id"], [row["call_id"] for row in base["registry"]["rows"]])
    partial.authorize_once(); partial.buffer_outputs(base["raw_rows"])
    try: partial.inspect_partial()
    except AccessViolation: pass
    scenario("partial_buffered_output_inspection", invalid, lambda p: p.__setitem__("access_report", partial.report()), "append-only access journal")
    scenario("frozen_source_drift", invalid, lambda p: p.__setitem__("source_manifest_observed", "0"*64), "isolated frozen-source manifest")
    scenario("stale_result_index", invalid, lambda p: p.__setitem__("payload_index_observed", "0"*64), "payload index")
    scenario("unknown_status_input", invalid, lambda p: p.__setitem__("requested_status", "UNKNOWN"), "release request")
    scenario("wrong_type_status_input", invalid, lambda p: p.__setitem__("requested_status", True), "release request")
    scenario("missing_seed", invalid, lambda p: p["expected_model_seed_identities"].pop(), "ordered seed manifest")
    scenario("duplicate_seed", invalid, lambda p: p["expected_model_seed_identities"].__setitem__(2,p["expected_model_seed_identities"][1]), "ordered seed manifest")
    scenario("reordered_seed", invalid, lambda p: p["expected_model_seed_identities"].reverse(), "ordered seed manifest")
    scenario("nonfinite_raw", invalid, lambda p: p["raw_rows"][0].__setitem__("applied_norm", float("nan")), "raw call")
    scenario("wrong_type_boolean", invalid, lambda p: p["raw_rows"][0].__setitem__("completed", 1), "raw call")

    seed0 = base["expected_model_seed_identities"][0]
    clean_target=lambda r: r["model_seed_identity"]==seed0 and r["method"]=="clean_task_decode" and r["axis"]=="blinded_within_run_target_composition"
    clean_parallel=lambda r: r["model_seed_identity"]==seed0 and r["method"]=="clean_task_decode" and r["axis"]=="blinded_within_run_parallel_composition"
    clean_all=lambda r: r["model_seed_identity"]==seed0 and r["method"]=="clean_task_decode"
    scenario("task_target_one_below", competence, lambda p:_force_correct(p,clean_target,"intervened_prediction","expected_answer",679), "raw clean predictions", {"correct":679,"denominator":800})
    scenario("task_parallel_one_below", competence, lambda p:_force_correct(p,clean_parallel,"intervened_prediction","expected_answer",679), "raw clean predictions", {"correct":679,"denominator":800})
    def worst_programme(p):
        pid=next(r["programme_id"] for r in p["registry"]["rows"] if clean_target(r)); _force_correct(p,lambda r: clean_target(r) and r["programme_id"]==pid,"intervened_prediction","expected_answer",11)
    scenario("worst_programme_one_below", competence, worst_programme, "raw clean predictions", {"correct":11,"denominator":16})
    for axis,selector in (("target",clean_target),("parallel",clean_parallel)):
        best=max(v["correct"] for v in base["shortcut_baselines"][f"clean_{axis}_axis"].values())
        scenario(f"shortcut_advantage_{axis}_one_below", competence, lambda p,s=selector,t=best+159:_force_correct(p,s,"intervened_prediction","expected_answer",t), "raw clean predictions versus frozen baseline", {"difference":159,"denominator":800})
    scenario("decode_one_below", no_eligible, lambda p:_force_correct(p,clean_all,"decoded_z2","expected_z2",1439), "raw subspace decoding", {"correct":1439,"denominator":1600})
    scenario("readout_ineligible", no_eligible, lambda p:p["readout_manifest"][0].__setitem__("eligible",False), "frozen readout manifest")

    primary=lambda stratum: lambda r: r["model_seed_identity"]==seed0 and r["method"]=="declared_iit_alignment" and r["pair_stratum"]==stratum
    scenario("answer_changing_iia_one_below", decode_only, lambda p:_force_correct(p,primary("A_answer_changing"),"intervened_prediction","expected_answer",135), "raw primary calls", {"correct":135,"denominator":160})
    for stratum,denominator,minimum in (("A_answer_changing",160,136),("B_outcome_collision",80,68),("D_same_observable_answer",80,68)):
        scenario(f"transport_{stratum}_one_below", negative, lambda p,s=stratum,t=minimum-1:_force_correct(p,primary(s),"decoded_z2","expected_z2",t), "raw subspace transport", {"correct":minimum-1,"denominator":denominator})
    def pooled_transport(p):
        _force_correct(p,primary("A_answer_changing"),"decoded_z2","expected_z2",151); _force_correct(p,primary("B_outcome_collision"),"decoded_z2","expected_z2",68); _force_correct(p,primary("D_same_observable_answer"),"decoded_z2","expected_z2",68)
    scenario("transport_pooled_one_below", negative, pooled_transport, "raw subspace transport", {"correct":287,"denominator":320})
    for stratum,denominator,minimum in (("A_answer_changing",160,136),("B_outcome_collision",80,68),("D_same_observable_answer",80,68)):
        scenario(f"descendant_{stratum}_one_below", negative, lambda p,s=stratum,t=minimum-1:_force_correct(p,primary(s),"intervened_z3","expected_z3",t), "raw descendant output", {"correct":minimum-1,"denominator":denominator})
    for stratum,denominator,minimum in (("A_answer_changing",160,152),("B_outcome_collision",80,76),("C_same_state",80,76),("D_same_observable_answer",80,76)):
        scenario(f"non_descendant_{stratum}_one_below", negative, lambda p,s=stratum,t=minimum-1:_force_correct(p,primary(s),"intervened_n3","expected_n3",t), "raw non-descendant output", {"correct":minimum-1,"denominator":denominator})
    scenario("same_state_one_below", negative, lambda p:_force_correct(p,primary("C_same_state"),"intervened_prediction","base_answer",75), "raw same-state calls", {"correct":75,"denominator":80})
    scenario("same_observable_answer_one_below", negative, lambda p:_force_correct(p,primary("D_same_observable_answer"),"intervened_prediction","expected_answer",63), "raw same-observable calls", {"correct":63,"denominator":80})
    control_ids=sorted({r["control_id"] for r in base["registry"]["rows"] if r["model_seed_identity"]==seed0 and r["control_id"] is not None})
    for control_id in control_ids:
        selector=lambda r,c=control_id:r["model_seed_identity"]==seed0 and r["control_id"]==c
        scenario(f"control_gap_{control_id}_one_below", negative, lambda p,s=selector:_force_correct(p,s,"intervened_prediction","expected_answer",113), "raw paired control calls", {"primary_correct":160,"control_correct":113,"difference":47,"denominator":160})
    group_selector=primary("A_answer_changing")
    scenario("base_accuracy_group_one_below", competence, lambda p:_force_correct(p,group_selector,"base_prediction","base_answer",135), "raw base predictions", {"correct":135,"denominator":160})
    scenario("source_accuracy_group_one_below", competence, lambda p:_force_correct(p,group_selector,"source_prediction","source_answer",135), "raw source predictions", {"correct":135,"denominator":160})

    report = {
        "schema": "exp33ae.preconfirmation_polarity.v1", "namespace": DEV_NAMESPACE, "claim_bearing": False,
        "scenario_count": len(scenarios), "passed_count": sum(row["passed"] for row in scenarios),
        "total_declared_scenarios": scenario_ordinal, "shard_index": shard_index, "shard_count": shard_count,
        "all_real_upstream": all(not row["direct_metric_dictionary_or_terminal_status_edit"] for row in scenarios),
        "scenarios": scenarios, "base_registry_sha256": sha256_file(fixture / "registry.jsonl"),
        "base_raw_sha256": sha256_file(fixture / "raw-results.jsonl"),
        "replay_source": "src/exp33ae/preconfirmation/polarity.py",
        "terminal_status_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    write_json(output, report)
    return report


def merge_polarity_shards(fixture: Path, shard_paths: list[Path], output: Path) -> dict:
    shards = [json.loads(path.read_text()) for path in shard_paths]
    scenarios = sorted((row for shard in shards for row in shard["scenarios"]), key=lambda row: row["ordinal"])
    ordinals = [row["ordinal"] for row in scenarios]
    if ordinals != list(range(len(scenarios))) or any(shard["total_declared_scenarios"] != len(scenarios) for shard in shards):
        raise RuntimeError("polarity shard coverage is missing, duplicated, or inconsistent")
    report = {
        "schema": "exp33ae.preconfirmation_polarity.v1", "namespace": DEV_NAMESPACE, "claim_bearing": False,
        "scenario_count": len(scenarios), "passed_count": sum(row["passed"] for row in scenarios),
        "all_real_upstream": all(not row["direct_metric_dictionary_or_terminal_status_edit"] for row in scenarios),
        "scenarios": scenarios, "shard_hashes": [sha256_file(path) for path in shard_paths],
        "base_registry_sha256": sha256_file(fixture / "registry.jsonl"),
        "base_raw_sha256": sha256_file(fixture / "raw-results.jsonl"),
        "replay_source": "src/exp33ae/preconfirmation/polarity.py",
        "terminal_status_materialized": False, "confirmation_objects_created": 0, "confirmation_objects_accessed": 0,
    }
    write_json(output, report)
    return report
