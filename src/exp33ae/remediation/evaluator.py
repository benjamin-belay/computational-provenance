"""Independent-byte-opening validator, metric evaluator, and typed aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import torch

from ..artifacts import tensor_sha256
from ..independent_oracle import oracle
from ..preconfirmation.allocator import allocate, causal_signature, shortcut_baselines
from ..preconfirmation.pairing import build_pairs
from ..spec import ALL_OPERATIONS
from .allocation import allocation_counts, load_allocation, primary_pair_counts
from .entropy import DevMasterEntropy
from .indexing import verify_prerelease_index
from .inference import execute_registry_segment
from .normative import DEV_NAMESPACE, POSITIVE_ANALOGUE, load_protocol, protocol_hash, verify_operative_manifest
from .runtime import environment_manifest, load_checkpoint
from .strict import expect_exact_keys, object_sha256, sha256_file, strict_load_json, strict_load_jsonl, StrictValidationError
from .training import primary_basis
from .validation import validate_raw, validate_registry


STATUS_COMPETENCE = "MODEL_COMPETENCE_FAILED"
STATUS_NO_ELIGIBLE = "NO_ELIGIBLE_CAUSAL_REPRESENTATION"
STATUS_DECODE_ONLY = "DECODE_ONLY_CAUSALITY_FAILED"
STATUS_NEGATIVE = "EXPERIMENT_NEGATIVE"
STATUS_INVALID = "EXPERIMENT_INVALID"


def _without_hash(value: dict, field: str) -> dict:
    copy = dict(value); copy.pop(field, None); return copy


def _verify_allocation(root: Path, output: Path, entropy: DevMasterEntropy, allocation: dict, pairs: list[dict], manifest: dict) -> None:
    directory = output / "allocation"
    expect_exact_keys(manifest, {"schema", "namespace", "claim_bearing", "stage_id", "protocol_sha256", "programme_count", "example_count", "pair_count", "partition_counts", "pair_strata", "production_independent_oracle_mismatches", "shortcut_baselines", "files", "development_exclusion_counts", "confirmation_objects_created", "confirmation_objects_accessed", "allocation_sha256"}, "allocation manifest")
    if manifest["allocation_sha256"] != object_sha256(_without_hash(manifest, "allocation_sha256")):
        raise StrictValidationError("ALLOCATION_BINDING", "manifest hash")
    for name, digest in manifest["files"].items():
        if sha256_file(directory / name) != digest:
            raise StrictValidationError("ALLOCATION_BINDING", name)
    expected = allocate(root, entropy, allocation_counts(), DEV_NAMESPACE)
    expected_pairs = build_pairs(expected, entropy, primary_pair_counts(), DEV_NAMESPACE)
    if [row for row in allocation["programmes"]] != expected["programmes"]:
        raise StrictValidationError("ALLOCATION_BINDING", "programme derivation")
    if [row.as_dict() for row in allocation["examples"]] != [row.as_dict() for row in expected["examples"]]:
        raise StrictValidationError("ALLOCATION_BINDING", "example derivation")
    if pairs != expected_pairs:
        raise StrictValidationError("ALLOCATION_BINDING", "pair derivation")
    if manifest["shortcut_baselines"] != shortcut_baselines(expected, entropy):
        raise StrictValidationError("ALLOCATION_BINDING", "shortcut baselines")
    if manifest["programme_count"] != 702 or manifest["example_count"] != 11232 or manifest["pair_count"] != 400:
        raise StrictValidationError("ALLOCATION_BINDING", "counts")


def _verify_representation(output: Path) -> dict:
    path = output / "representation.json"
    value = strict_load_json(path)
    expect_exact_keys(value, {"schema", "namespace", "claim_bearing", "protocol_sha256", "target", "layer", "position", "rank", "strength", "basis", "basis_sha256", "decoder", "state_codes", "secondary_methods", "adjudicating_decoder", "representation_sha256"}, "representation")
    if value["representation_sha256"] != object_sha256(_without_hash(value, "representation_sha256")):
        raise StrictValidationError("BASIS_DECODER_BINDING", "representation hash")
    basis = torch.tensor(value["basis"], dtype=torch.float32)
    if basis.shape != (64, 2) or tensor_sha256(basis) != value["basis_sha256"] or not torch.equal(basis, primary_basis()):
        raise StrictValidationError("BASIS_DECODER_BINDING", "primary basis")
    if value["decoder"] != {"kind": "registered_coordinate_argmax", "classes": 2, "coordinates": [0, 1]}:
        raise StrictValidationError("BASIS_DECODER_BINDING", "decoder")
    return value


def _verify_controls(output: Path, seeds: list[int], representation: dict) -> list[dict]:
    manifests = []
    primary = torch.tensor(representation["basis"], dtype=torch.float32)
    for order, seed in enumerate(seeds):
        path = output / "controls" / f"model-{order}.json"
        value = strict_load_json(path)
        expect_exact_keys(value, {"schema", "namespace", "claim_bearing", "stage_id", "protocol_sha256", "model_order", "model_seed_identity", "controls", "norm_tolerance_absolute", "norm_tolerance_relative", "control_manifest_sha256"}, f"control manifest {order}")
        if value["control_manifest_sha256"] != object_sha256(_without_hash(value, "control_manifest_sha256")) or value["model_order"] != order or value["model_seed_identity"] != seed:
            raise StrictValidationError("CONTROL_BINDING", f"control manifest {order}")
        controls = value["controls"]
        if Counter(row["family"] for row in controls) != Counter({"random_direction": 4, "orthogonal_direction": 4, "alternate_location": 3, "alternate_early_layer": 1}):
            raise StrictValidationError("CONTROL_BINDING", f"control counts {order}")
        for row in controls:
            expect_exact_keys(row, {"control_id", "family", "index", "layer", "position", "rank", "strength", "basis", "basis_sha256"}, "control row")
            basis = torch.tensor(row["basis"], dtype=torch.float32)
            if basis.shape != (64, 2) or tensor_sha256(basis) != row["basis_sha256"]:
                raise StrictValidationError("CONTROL_BINDING", f"control basis {row['control_id']}")
            if row["family"] == "orthogonal_direction" and float((primary.T @ basis).abs().max()) > 1e-5:
                raise StrictValidationError("CONTROL_BINDING", "orthogonal control")
            if row["family"] == "alternate_early_layer" and (row["layer"], row["position"]) != (0, 14):
                raise StrictValidationError("CONTROL_BINDING", "early layer site")
        manifests.append(value)
    return manifests


def _verify_training(output: Path, seeds: list[int], operative: dict, environment: dict) -> list[dict]:
    values = []
    for order, seed in enumerate(seeds):
        path = output / "training" / f"model-{order}.json"
        value = strict_load_json(path)
        expect_exact_keys(value, {"schema", "namespace", "claim_bearing", "protocol_sha256", "model_order", "model_seed_identity", "batch_stream_seed", "interchange_stream_seed", "initial_tensor_state_sha256", "final_tensor_state_sha256", "checkpoint_relative_path", "checkpoint_file_sha256", "checkpoint_bytes", "steps", "batch_size", "interchange_pairs", "final_step_only", "curve", "operative_manifest_sha256", "environment_sha256", "scientific_manifest_sha256"}, f"training manifest {order}")
        if value["scientific_manifest_sha256"] != object_sha256(_without_hash(value, "scientific_manifest_sha256")):
            raise StrictValidationError("TRAINING_BINDING", f"training manifest {order}")
        if value["model_order"] != order or value["model_seed_identity"] != seed or value["steps"] != 1400 or value["interchange_pairs"] != 358400:
            raise StrictValidationError("TRAINING_BINDING", f"training identity {order}")
        if value["batch_stream_seed"] == value["interchange_stream_seed"]:
            raise StrictValidationError("TRAINING_BINDING", "stream collision")
        if value["operative_manifest_sha256"] != operative["manifest_sha256"] or value["environment_sha256"] != environment["manifest_sha256"]:
            raise StrictValidationError("TRAINING_BINDING", "source/environment")
        load_checkpoint(output / value["checkpoint_relative_path"], {
            "model_order": order, "model_seed_identity": seed,
            "operative_manifest_sha256": operative["manifest_sha256"], "environment_sha256": environment["manifest_sha256"],
            "tensor_state_sha256": value["final_tensor_state_sha256"], "checkpoint_file_sha256": value["checkpoint_file_sha256"],
        })
        values.append(value)
    return values


def _verify_registry_semantics(rows, examples, pairs, controls, training, representation, allocation_manifest, operative, environment, output):
    by_id = {row.example_id: row for row in examples}
    pair_by_id = {row["pair_id"]: row for row in pairs}
    control_by_model = [{row["control_id"]: row for row in value["controls"]} for value in controls]
    for row in rows:
        base = by_id.get(row["base_id"]); source = by_id.get(row["source_id"])
        if base is None or source is None:
            raise StrictValidationError("REGISTRY_BINDING", "unknown example")
        if row["programme_id"] != base.programme_id or row["causal_signature"] != base.causal_signature or row["axis"] != base.axis:
            raise StrictValidationError("REGISTRY_BINDING", "programme/example identity")
        for side, example in (("base", base), ("source", source)):
            for field in ("z2", "z3", "n3", "answer"):
                expected = example.answer if field == "answer" else getattr(example, field)
                if row[f"{side}_{field}"] != expected:
                    raise StrictValidationError("REGISTRY_BINDING", f"{side} value")
        if row["method"] == "clean_task_decode":
            if base.example_id != source.example_id or row["expected_answer"] != base.answer or row["expected_z2"] != base.z2 or row["strength"] != 0.0:
                raise StrictValidationError("REGISTRY_BINDING", "clean call")
        else:
            pair = pair_by_id.get(row["pair_id"])
            if pair is None or pair["base_id"] != base.example_id or pair["source_id"] != source.example_id:
                raise StrictValidationError("REGISTRY_BINDING", "pair identity")
            for field in ("z2", "z3", "n3", "answer"):
                if row[f"expected_{field}"] != pair[f"expected_{field}"]:
                    raise StrictValidationError("REGISTRY_BINDING", "oracle expectation")
        order = row["model_order"]; t = training[order]
        if row["checkpoint_file_sha256"] != t["checkpoint_file_sha256"] or row["checkpoint_tensor_sha256"] != t["final_tensor_state_sha256"] or row["training_manifest_sha256"] != t["scientific_manifest_sha256"]:
            raise StrictValidationError("CHECKPOINT_FILE_BINDING", "registry/checkpoint")
        if row["operative_manifest_sha256"] != operative["manifest_sha256"] or row["environment_sha256"] != environment["manifest_sha256"] or row["allocation_sha256"] != allocation_manifest["allocation_sha256"] or row["representation_sha256"] != representation["representation_sha256"]:
            raise StrictValidationError("REGISTRY_BINDING", "operative binding")
        if row["control_id"] is None:
            expected_basis = representation["basis_sha256"]
        else:
            control = control_by_model[order].get(row["control_id"])
            if control is None or (row["method"], row["layer"], row["position"], row["rank"], row["strength"]) != (control["family"], control["layer"], control["position"], control["rank"], control["strength"]):
                raise StrictValidationError("CONTROL_BINDING", "control plan")
            expected_basis = control["basis_sha256"]
        if row["basis_sha256"] != expected_basis:
            raise StrictValidationError("CONTROL_BINDING", "control basis plan")


def _metrics_for_seed(plans, raw, shortcuts):
    observed = {row["call_id"]: row for row in raw}
    clean = [row for row in plans if row["method"] == "clean_task_decode"]
    primary = [row for row in plans if row["method"] == "declared_iit_alignment"]
    controls = [row for row in plans if row["control_id"] is not None]
    def count(items, predicate): return sum(predicate(plan, observed[plan["call_id"]]) for plan in items)
    axes = {axis: [row for row in clean if row["axis"] == axis] for axis in ("blinded_within_run_target_composition", "blinded_within_run_parallel_composition")}
    programme_correct, programme_total = Counter(), Counter()
    for plan in clean:
        programme_total[plan["programme_id"]] += 1
        programme_correct[plan["programme_id"]] += observed[plan["call_id"]]["intervened_prediction"] == plan["expected_answer"]
    strata = {name: [row for row in primary if row["pair_stratum"] == name] for name in ("A_answer_changing", "B_outcome_collision", "C_same_state", "D_same_observable_answer")}
    groups = defaultdict(list)
    for plan in plans: groups[(plan["method"], plan["control_id"], plan["pair_stratum"])].append(plan)
    values = {
        "task_target": count(axes["blinded_within_run_target_composition"], lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "task_parallel": count(axes["blinded_within_run_parallel_composition"], lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "worst_programme": min(programme_correct.values()),
        "decode": count(clean, lambda p,r:r["decoded_z2"]==p["expected_z2"]),
        "iia_A": count(strata["A_answer_changing"], lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "transport": {name: count(rows, lambda p,r:r["decoded_z2"]==p["expected_z2"]) for name,rows in strata.items()},
        "descendant": {name: count(rows, lambda p,r:r["intervened_z3"]==p["expected_z3"]) for name,rows in strata.items()},
        "non_descendant": {name: count(rows, lambda p,r:r["intervened_n3"]==p["expected_n3"]) for name,rows in strata.items()},
        "same_state": count(strata["C_same_state"], lambda p,r:r["intervened_prediction"]==p["base_answer"]),
        "iia_D": count(strata["D_same_observable_answer"], lambda p,r:r["intervened_prediction"]==p["expected_answer"]),
        "control_iia": {},
        "base_groups": {}, "source_groups": {}, "group_denominators": {},
        "shortcut_target": max(value["correct"] for value in shortcuts["clean_target_axis"].values()),
        "shortcut_parallel": max(value["correct"] for value in shortcuts["clean_parallel_axis"].values()),
    }
    for control_id in sorted({row["control_id"] for row in controls}):
        rows = [row for row in controls if row["control_id"] == control_id]
        values["control_iia"][control_id] = count(rows, lambda p,r:r["intervened_prediction"]==p["expected_answer"])
    for key, rows in groups.items():
        label = repr(key); values["group_denominators"][label] = len(rows)
        values["base_groups"][label] = count(rows, lambda p,r:r["base_prediction"]==p["base_answer"])
        values["source_groups"][label] = count(rows, lambda p,r:r["source_prediction"]==p["source_answer"])
    return values


def _seed_status(metrics):
    competence = (
        metrics["task_target"] >= 680 and metrics["task_parallel"] >= 680 and metrics["worst_programme"] >= 12
        and metrics["task_target"] - metrics["shortcut_target"] >= 160
        and metrics["task_parallel"] - metrics["shortcut_parallel"] >= 160
        and all(20*value >= 17*metrics["group_denominators"][key] for key,value in metrics["base_groups"].items())
        and all(20*value >= 17*metrics["group_denominators"][key] for key,value in metrics["source_groups"].items())
    )
    eligible = metrics["decode"] >= 1440
    causal = metrics["iia_A"] >= 136
    other = (
        metrics["transport"]["A_answer_changing"] >= 136 and metrics["transport"]["B_outcome_collision"] >= 68 and metrics["transport"]["D_same_observable_answer"] >= 68
        and sum(metrics["transport"][name] for name in ("A_answer_changing","B_outcome_collision","D_same_observable_answer")) >= 288
        and metrics["descendant"]["A_answer_changing"] >= 136 and metrics["descendant"]["B_outcome_collision"] >= 68 and metrics["descendant"]["D_same_observable_answer"] >= 68
        and metrics["non_descendant"]["A_answer_changing"] >= 152
        and all(metrics["non_descendant"][name] >= 76 for name in ("B_outcome_collision","C_same_state","D_same_observable_answer"))
        and metrics["same_state"] >= 76 and metrics["iia_D"] >= 64
        and len(metrics["control_iia"]) == 12 and all(metrics["iia_A"] - value >= 48 for value in metrics["control_iia"].values())
    )
    if not competence: return STATUS_COMPETENCE
    if not eligible: return STATUS_NO_ELIGIBLE
    if not causal: return STATUS_DECODE_ONLY
    if not other: return STATUS_NEGATIVE
    return POSITIVE_ANALOGUE


def evaluate(output: Path, root: Path, access_snapshot: dict) -> dict:
    try:
        if access_snapshot["state"]["state"] != "BUFFER_COMPLETE" or access_snapshot["state"]["buffered_calls"] != 11760:
            raise StrictValidationError("ACCESS_STATE", "buffer not complete")
        entropy_record = strict_load_json(output / "entropy.json")
        expect_exact_keys(entropy_record, {"schema", "namespace", "stage_id", "claim_bearing", "protocol_sha256", "dev_only_master_hex", "commitment", "model_seed_identities", "confirmation_master_entropy_created", "confirmation_derived_objects_created", "record_sha256"}, "entropy record")
        if entropy_record["schema"] != "exp33ae.dev_only_entropy_record.v2" or entropy_record["namespace"] != DEV_NAMESPACE or entropy_record["claim_bearing"] is not False or entropy_record["protocol_sha256"] != protocol_hash() or entropy_record["confirmation_master_entropy_created"] is not False or entropy_record["confirmation_derived_objects_created"] != 0 or entropy_record["record_sha256"] != object_sha256(_without_hash(entropy_record, "record_sha256")):
            raise StrictValidationError("STAGE_GUARD", "entropy record")
        entropy = DevMasterEntropy(entropy_record["stage_id"], bytes.fromhex(entropy_record["dev_only_master_hex"]))
        if entropy.commitment != entropy_record["commitment"] or list(entropy.model_seeds()) != entropy_record["model_seed_identities"]:
            raise StrictValidationError("STAGE_GUARD", "entropy commitment/seeds")
        operative = strict_load_json(output / "operative-source-manifest.json")
        faults = verify_operative_manifest(root, operative)
        if faults: raise StrictValidationError(faults[0], "operative source")
        environment = strict_load_json(output / "environment.json")
        if environment["manifest_sha256"] != object_sha256(_without_hash(environment, "manifest_sha256")) or environment != environment_manifest():
            raise StrictValidationError("ENVIRONMENT_BINDING", "runtime environment")
        allocation, pairs, allocation_manifest = load_allocation(output)
        _verify_allocation(root, output, entropy, allocation, pairs, allocation_manifest)
        representation = _verify_representation(output)
        seeds = list(entropy.model_seeds())
        controls = _verify_controls(output, seeds, representation)
        training = _verify_training(output, seeds, operative, environment)
        registry_manifest = strict_load_json(output / "registry" / "manifest.json")
        registry = strict_load_jsonl(output / "registry" / "registry.jsonl")
        if registry_manifest["registry_manifest_sha256"] != object_sha256(_without_hash(registry_manifest, "registry_manifest_sha256")) or registry_manifest["registry_file_sha256"] != sha256_file(output / "registry" / "registry.jsonl"):
            raise StrictValidationError("REGISTRY_BINDING", "registry manifest/file")
        validate_registry(registry, registry_manifest, seeds)
        _verify_registry_semantics(registry, allocation["examples"], pairs, controls, training, representation, allocation_manifest, operative, environment, output)
        raw = access_snapshot["buffer_rows"]
        validate_raw(raw, registry)
        # Raw prediction and activation rows are not trusted merely because
        # their internal hashes are coherent.  Re-run the frozen checkpoints,
        # hooks, bases, and plans and require byte-canonical row equality.
        examples_by_id = {row.example_id: row for row in allocation["examples"]}
        reconstructed = []
        for start in range(0, len(registry), 256):
            reconstructed.extend(execute_registry_segment(output, registry[start:start + 256], examples_by_id, training))
        if reconstructed != raw:
            raise StrictValidationError("RAW_BINDING", "independent frozen-inference reconstruction")
        index = verify_prerelease_index(output)
        if not index["matches"]:
            raise StrictValidationError("INDEX_BINDING", "prerelease index")
        metrics_by_seed, seed_statuses = {}, []
        for order, seed in enumerate(seeds):
            plans = registry[order*3920:(order+1)*3920]
            observations = raw[order*3920:(order+1)*3920]
            metrics = _metrics_for_seed(plans, observations, allocation_manifest["shortcut_baselines"])
            status = _seed_status(metrics)
            metrics_by_seed[str(seed)] = metrics
            seed_statuses.append(status)
        if STATUS_COMPETENCE in seed_statuses: aggregate = STATUS_COMPETENCE
        elif STATUS_NO_ELIGIBLE in seed_statuses: aggregate = STATUS_NO_ELIGIBLE
        elif STATUS_DECODE_ONLY in seed_statuses: aggregate = STATUS_DECODE_ONLY
        elif STATUS_NEGATIVE in seed_statuses: aggregate = STATUS_NEGATIVE
        elif seed_statuses == [POSITIVE_ANALOGUE] * 3: aggregate = POSITIVE_ANALOGUE
        else: aggregate = STATUS_INVALID
        return {
            "schema": "exp33ae.dev_only_derived_status.v2", "namespace": DEV_NAMESPACE,
            "claim_bearing": False, "terminal_confirmation_status_materialized": False,
            "status": aggregate, "seed_statuses": seed_statuses, "model_seed_identities": seeds,
            "metrics_by_seed": metrics_by_seed, "invalidity_reasons": [], "registry_calls": 11760,
            "prerelease_index_sha256": index["observed"],
        }
    except StrictValidationError as error:
        return {
            "schema": "exp33ae.dev_only_derived_status.v2", "namespace": DEV_NAMESPACE,
            "claim_bearing": False, "terminal_confirmation_status_materialized": False,
            "status": STATUS_INVALID, "seed_statuses": [], "model_seed_identities": [], "metrics_by_seed": {},
            "invalidity_reasons": [error.reason], "invalidity_detail": error.detail,
        }
    except Exception as error:
        return {
            "schema": "exp33ae.dev_only_derived_status.v2", "namespace": DEV_NAMESPACE,
            "claim_bearing": False, "terminal_confirmation_status_materialized": False,
            "status": STATUS_INVALID, "seed_statuses": [], "model_seed_identities": [], "metrics_by_seed": {},
            "invalidity_reasons": ["INTERNAL_VALIDATION_EXCEPTION"], "invalidity_detail": type(error).__name__,
        }
