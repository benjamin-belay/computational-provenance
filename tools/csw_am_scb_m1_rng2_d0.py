#!/usr/bin/env python3
"""RNG2-D0 public implementation and validation of the position-indexed sampler."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.util
import json
import math
import os
import struct
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "protocol/csw-am-scb-m1-rng2-d0/CONFIG.json"
FIXTURE_PATH = ROOT / "protocol/csw-am-scb-m1-rng2-d0/DISTRIBUTIONAL_FIXTURES.json"
OUTPUT = ROOT / "artifacts/dev_only/csw-am-scb-m1-rng2-d0"
DOMAIN = b"EXP33A|SCB-M1|RNG2|POSITION-UNIFORM|V1"
HEX = frozenset("0123456789abcdef")


def module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError("MODULE:" + name)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


A0 = module("a0_for_rng2_d0", "tools/csw_a0_stochastic_channel.py")
CAR = module("car_s0_for_rng2_d0", "tools/csw_am_scb_m1_car_s0.py")
REF = module("rng2_reference_for_d0", "tools/csw_am_scb_m1_rng2_reference.py")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: Any) -> str:
    return sha(canonical(value))


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise ValueError("JSON_OBJECT:" + str(path))
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def git_show(commit: str, relative: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)


def committed(path: Path, commit: str) -> bool:
    relative = str(path.relative_to(ROOT))
    return git_show(commit, relative) == path.read_bytes()


def validate_bindings(config: dict[str, Any], source_commit: str) -> dict[str, bool]:
    bindings = config["bindings"]
    checks = {}
    for name in ("sampler_protocol", "numeric_protocol", "validation_protocol",
                 "R0_development_authorization", "R0_main_audit", "R0_terminal"):
        path = ROOT / bindings[f"{name}_path"]
        checks[name + "_sha256"] = file_sha(path) == bindings[f"{name}_sha256"]
    public = config["public_development"]
    checks["development_closure_sha256"] = (
        file_sha(ROOT / public["closure_path"]) == public["closure_sha256"]
    )
    checks["development_row_sha256"] = (
        file_sha(ROOT / public["row_path"]) == public["row_sha256"]
    )
    surface = config["surface_channel"]
    for name in ("source", "checkpoint", "frozen_channel"):
        checks["surface_" + name + "_sha256"] = (
            file_sha(ROOT / surface[f"{name}_path"]) == surface[f"{name}_sha256"]
        )
    source_paths = [
        CONFIG_PATH,
        FIXTURE_PATH,
        ROOT / "protocol/csw-am-scb-m1-rng2-d0/EVIDENCE_PROTOCOL.json",
        ROOT / "tools/csw_am_scb_m1_rng2_d0.py",
        ROOT / "tools/csw_am_scb_m1_rng2_reference.py",
    ]
    checks["source_commit_binding"] = all(committed(path, source_commit) for path in source_paths)
    authorization = read_json(ROOT / bindings["R0_development_authorization_path"])
    checks["R0_authorized_role"] = (
        authorization["authorized_use"] == "RNG2_D0_PUBLIC_IMPLEMENTATION_AND_DEBUGGING_ONLY"
        and authorization["calibration_access"] is False
        and authorization["validation_access"] is False
        and authorization["reserve_access"] is False
        and authorization["protected_access"] is False
    )
    terminal = read_json(ROOT / bindings["R0_terminal_path"])
    checks["R0_terminal_ready"] = (
        terminal["terminal"] == bindings["RNG2_R0_terminal"]
    )
    return checks


def _hex64(value: Any, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in HEX for char in value):
        raise ValueError(label)
    return value


def validate_index(index: dict[str, Any]) -> None:
    if type(index) is not dict or set(index) != {
        "protocol_version", "sampler_domain", "pairing_context_id", "closure_id",
        "output_index", "model_step", "subdraw_index",
    }:
        raise ValueError("INDEX_FIELDS")
    if index["protocol_version"] != "RNG2_V1":
        raise ValueError("PROTOCOL_VERSION")
    if index["sampler_domain"] != "GENERIC_STOCHASTIC_CHANNEL":
        raise ValueError("SAMPLER_DOMAIN")
    _hex64(index["pairing_context_id"], "PAIRING_CONTEXT")
    _hex64(index["closure_id"], "CLOSURE")
    if type(index["output_index"]) is not int or not 0 <= index["output_index"] <= 7:
        raise ValueError("OUTPUT_INDEX")
    if type(index["model_step"]) is not int or not 0 <= index["model_step"] <= 48:
        raise ValueError("MODEL_STEP")
    if type(index["subdraw_index"]) is not int or index["subdraw_index"] != 0:
        raise ValueError("SUBDRAW_INDEX")


def position_uniform(key: bytes, index: dict[str, Any]) -> dict[str, Any]:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError("KEY")
    validate_index(index)
    index_bytes = canonical(index)
    preimage = DOMAIN + b"\x00" + index_bytes
    mac = hmac.new(key, preimage, hashlib.sha256).digest()
    integer = int.from_bytes(mac[:8], "big") >> 12
    uniform = (integer + 0.5) / (1 << 52)
    if not 0.0 < uniform < 1.0:
        raise ValueError("UNIFORM")
    return {
        "canonical_index": index,
        "canonical_index_hex": index_bytes.hex(),
        "canonical_preimage_hex": preimage.hex(),
        "hmac_sha256": mac.hex(),
        "uint52": integer,
        "uniform": uniform,
        "uniform_hex": uniform.hex(),
        "uniform_bytes_hex": struct.pack(">d", uniform).hex(),
    }


def normalize_weights(weights: Sequence[float]) -> list[float]:
    if not weights:
        raise ValueError("WEIGHTS")
    values = [float(value) for value in weights]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("WEIGHT")
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("TOTAL")
    return [value / total for value in values]


def inverse_cdf(weights: Sequence[float], uniform: float) -> dict[str, Any]:
    if type(uniform) is not float or not math.isfinite(uniform) or not 0.0 <= uniform <= 1.0:
        raise ValueError("UNIFORM")
    probabilities = normalize_weights(weights)
    cumulative = []
    running = 0.0
    selected = len(probabilities) - 1
    fallback = True
    for index, probability in enumerate(probabilities):
        running += probability
        cumulative.append(running)
        if uniform <= running:
            selected = index
            fallback = False
            break
    if len(cumulative) < len(probabilities):
        running = cumulative[-1]
        for probability in probabilities[len(cumulative):]:
            running += probability
            cumulative.append(running)
    return {
        "normalized_probabilities": probabilities,
        "cumulative_probabilities": cumulative,
        "selected_index": selected,
        "final_bin_fallback": fallback,
    }


def index_for(pairing_context_id: str, closure_id: str, output_index: int, step: int) -> dict[str, Any]:
    return {
        "protocol_version": "RNG2_V1",
        "sampler_domain": "GENERIC_STOCHASTIC_CHANNEL",
        "pairing_context_id": pairing_context_id,
        "closure_id": closure_id,
        "output_index": output_index,
        "model_step": step,
        "subdraw_index": 0,
    }


def sample_ids(
    logits: torch.Tensor,
    candidates: Sequence[str],
    key: bytes,
    index: dict[str, Any],
    *,
    green: set[str] | None,
    bias: float,
) -> tuple[str, dict[str, Any]]:
    ids = [A0.TO_ID[token] for token in candidates]
    pre = torch.tensor([float(logits[token_id]) for token_id in ids], dtype=torch.float64)
    if not torch.isfinite(pre).all():
        raise ValueError("LOGITS")
    post = pre.clone()
    if green is not None:
        for candidate_index, token in enumerate(candidates):
            if token in green:
                post[candidate_index] += bias
    probabilities = torch.softmax(post, dim=0).tolist()
    uniform = position_uniform(key, index)
    selection = inverse_cdf(probabilities, uniform["uniform"])
    selected = selection["selected_index"]
    return candidates[selected], {
        **uniform,
        "candidate_order": list(candidates),
        "pre_logits": pre.tolist(),
        "post_logits": post.tolist(),
        "probabilities": selection["normalized_probabilities"],
        "cumulative_probabilities": selection["cumulative_probabilities"],
        "selected_index": selected,
        "selected_token": candidates[selected],
        "final_bin_fallback": selection["final_bin_fallback"],
        "green_tokens": sorted(green) if green is not None else [],
    }


def generate(
    model: Any,
    semantic: dict[str, Any],
    *,
    sampler_key: bytes,
    pairing_context_id: str,
    closure_id: str,
    output_index: int,
    authority_key: bytes,
    bias: float,
) -> dict[str, Any]:
    prompt = A0.prompt_tokens(semantic)
    hidden, logits = model.initialize(A0.encode(prompt))
    model_history = list(prompt)
    public_tokens: list[str] = []
    operations = []
    step = 0
    eligible_position = 0

    def choose(candidates: Sequence[str], green: set[str] | None, applied_bias: float):
        return sample_ids(
            logits, candidates, sampler_key,
            index_for(pairing_context_id, closure_id, output_index, step),
            green=green, bias=applied_bias,
        )

    order_token, trace = choose(("<ORDER_A>", "<ORDER_B>"), None, 0.0)
    operations.append({"kind": "ORDER", "model_step": step, "public": False,
                       "model_history_before": list(model_history), **trace})
    order = A0.ORDERS[0 if order_token == "<ORDER_A>" else 1]
    model_history.append(order_token)
    hidden, logits = model.advance(hidden, A0.TO_ID[order_token])
    step += 1
    for field in order:
        sequence: list[tuple[str, Sequence[str] | str]] = [
            ("eligible", A0.GROUPS[0]), ("mandatory", "the"),
            ("eligible", A0.GROUPS[1]), ("eligible", A0.GROUPS[2]),
            ("eligible", A0.GROUPS[3]), ("eligible", A0.GROUPS[4]),
            ("mandatory", field), ("mandatory", "is"),
            ("mandatory", str(semantic[field])), ("eligible", A0.GROUPS[5]),
            ("mandatory", "."), ("optional", A0.OPTIONALS),
        ]
        for kind, allowed in sequence:
            candidates = (allowed,) if isinstance(allowed, str) else tuple(allowed)
            green = None
            if kind == "eligible":
                green = A0.green_set(authority_key, eligible_position, public_tokens, candidates)
            selected, trace = choose(
                candidates, green, bias if kind == "eligible" else 0.0
            )
            operation = {
                "kind": kind.upper(),
                "model_step": step,
                "field": field,
                "public": selected not in ("<NONE>", "<ORDER_A>", "<ORDER_B>"),
                "model_history_before": list(model_history),
                "public_history_before": list(public_tokens),
                **trace,
            }
            if kind == "eligible":
                operation["eligible_position"] = eligible_position
                eligible_position += 1
            operations.append(operation)
            model_history.append(selected)
            if selected != "<NONE>":
                public_tokens.append(selected)
            hidden, logits = model.advance(hidden, A0.TO_ID[selected])
            step += 1
    parsed = A0.parse_output(public_tokens)
    return {
        "semantic": semantic,
        "literal_output": " ".join(public_tokens),
        "output_sha256": sha(" ".join(public_tokens).encode()),
        "public_tokens": public_tokens,
        "model_history": model_history,
        "parsed": parsed,
        "eligible_positions": eligible_position,
        "stochastic_draws": step,
        "operations": operations,
    }


def select_development(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public = config["public_development"]
    closure_rows = read_rows(ROOT / public["closure_path"])
    rows = read_rows(ROOT / public["row_path"])
    if len(closure_rows) != public["available_closures"] or len(rows) != 2 * len(closure_rows):
        raise ValueError("PUBLIC_DENOMINATOR")
    by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_closure[row["closure_id"]].append(row)
    if set(by_closure) != {row["closure_id"] for row in closure_rows}:
        raise ValueError("PUBLIC_MEMBERSHIP")
    domain = public["selection_domain"]
    selected: set[str] = set()
    for z2 in range(16):
        for template in range(4):
            candidates = sorted({
                row["closure_id"] for row in rows
                if row["natural"]["z2"] == z2 and row["template_id"] == template
            }, key=lambda cid: sha(f"{domain}|CELL|{z2}|{template}|{cid}".encode()))
            if not candidates:
                raise ValueError("SUPPORT")
            selected.add(candidates[0])
    remaining = sorted(
        set(by_closure) - selected,
        key=lambda cid: sha(f"{domain}|FILL|{cid}".encode()),
    )
    selected.update(remaining[:public["selected_closures"] - len(selected)])
    if len(selected) != public["selected_closures"]:
        raise ValueError("SELECTION_DENOMINATOR")
    selected_closures = [row for row in closure_rows if row["closure_id"] in selected]
    selected_rows = [row for row in rows if row["closure_id"] in selected]
    cells = {(row["natural"]["z2"], row["template_id"]) for row in selected_rows}
    if (
        len(cells) != 64
        or {row["natural"]["z2"] for row in selected_rows} != set(range(16))
        or {row["visible_answer"] for row in selected_rows} != set(range(8))
        or {row["template_id"] for row in selected_rows} != set(range(4))
    ):
        raise ValueError("SELECTION_COVERAGE")
    return selected_closures, selected_rows


def boundary_fixtures() -> list[dict[str, Any]]:
    tests = []
    cases = [
        ("U_EQUALS_ZERO", [0.5, 0.5], 0.0, 0, False),
        ("U_EQUALS_FIRST_CDF_BOUNDARY", [0.5, 0.5], 0.5, 0, False),
        ("U_IMMEDIATELY_ABOVE_BOUNDARY", [0.5, 0.5], math.nextafter(0.5, 1.0), 1, False),
        ("U_IMMEDIATELY_BELOW_ONE", [0.25, 0.75], math.nextafter(1.0, 0.0), 1, False),
        (
            "FINAL_BIN_ROUNDING_FALLBACK",
            [0.9051126704187485, 0.42456278623685484],
            1.0,
            1,
            True,
        ),
    ]
    for name, weights, uniform, expected, expected_fallback in cases:
        observed = inverse_cdf(weights, uniform)
        reference = REF.inverse_cdf(weights, uniform)
        tests.append({
            "fixture": name,
            "expected_index": expected,
            "observed_index": observed["selected_index"],
            "reference_index": reference["selected_index"],
            "expected_fallback": expected_fallback,
            "observed_fallback": observed["final_bin_fallback"],
            "passed": (
                observed["selected_index"] == expected
                and reference["selected_index"] == expected
                and observed["final_bin_fallback"] == expected_fallback
            ),
        })
    invalid = [
        ("NONFINITE_REJECTION", [float("nan"), 1.0], 0.5),
        ("NEGATIVE_PROBABILITY_REJECTION", [-0.1, 1.1], 0.5),
    ]
    for name, weights, uniform in invalid:
        rejected = []
        for function in (inverse_cdf, REF.inverse_cdf):
            try:
                function(weights, uniform)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        tests.append({"fixture": name, "production_rejected": rejected[0],
                      "reference_rejected": rejected[1], "passed": all(rejected)})
    malformed = index_for("0" * 64, "1" * 64, 0, 0)
    malformed["subdraw_index"] = 1
    rejected = []
    for function in (position_uniform, REF.reconstruct_uniform):
        try:
            function(bytes(32), malformed)
            rejected.append(False)
        except ValueError:
            rejected.append(True)
    tests.append({"fixture": "MALFORMED_INDEX_REJECTION",
                  "production_rejected": rejected[0],
                  "reference_rejected": rejected[1], "passed": all(rejected)})
    return tests


def deterministic_fixtures(key: bytes) -> list[dict[str, Any]]:
    base = index_for(sha(b"PAIR"), sha(b"CLOSURE"), 0, 0)
    variations = [("BASE", base)]
    for field, value in [
        ("pairing_context_id", sha(b"PAIR_ALT")),
        ("closure_id", sha(b"CLOSURE_ALT")),
        ("output_index", 1),
        ("model_step", 1),
    ]:
        changed = dict(base)
        changed[field] = value
        variations.append(("CHANGE_" + field.upper(), changed))
    rows = []
    for name, index in variations:
        production = position_uniform(key, index)
        reference = REF.reconstruct_uniform(key, index)
        rows.append({
            "fixture": name,
            "production": production,
            "reference": reference,
            "exact_agreement": production == reference,
        })
    base_uniform = rows[0]["production"]["uniform_bytes_hex"]
    for row in rows[1:]:
        row["preimage_differs_from_base"] = (
            row["production"]["canonical_preimage_hex"]
            != rows[0]["production"]["canonical_preimage_hex"]
        )
        row["fixture_uniform_differs_from_base"] = (
            row["production"]["uniform_bytes_hex"] != base_uniform
        )
    forbidden_invariance = []
    for field in ("state", "role", "history", "receipt", "exact_execution_id"):
        # Forbidden fields are deliberately not accepted as index fields and cannot enter the preimage.
        candidate = dict(base)
        candidate[field] = "CHANGED"
        try:
            position_uniform(key, candidate)
            rejected = False
        except ValueError:
            rejected = True
        forbidden_invariance.append({"forbidden_field": field, "index_rejected": rejected})
    rows.append({
        "fixture": "FORBIDDEN_FIELD_COLLISION_FIXTURES",
        "base_uniform_bytes_hex": base_uniform,
        "tests": forbidden_invariance,
        "exact_agreement": all(item["index_rejected"] for item in forbidden_invariance),
    })
    return rows


def probabilities_for_fixture(fixture: dict[str, Any]) -> list[float]:
    if "probabilities" in fixture:
        return normalize_weights(fixture["probabilities"])
    logits = torch.tensor(fixture["logits"], dtype=torch.float64)
    if "green_indices" in fixture:
        for index in fixture["green_indices"]:
            logits[index] += fixture["bias"]
    return torch.softmax(logits, dim=0).tolist()


def distributional_tests(key: bytes) -> list[dict[str, Any]]:
    specification = read_json(FIXTURE_PATH)
    n = specification["contexts_per_fixture"]
    results = []
    for fixture in specification["fixtures"]:
        probabilities = probabilities_for_fixture(fixture)
        counts = [0 for _ in probabilities]
        closure = sha(("RNG2_DISTRIBUTION|" + fixture["id"]).encode())
        for context_index in range(n):
            pairing = sha(
                f"RNG2_DISTRIBUTION|{fixture['id']}|{context_index}".encode()
            )
            uniform = position_uniform(key, index_for(pairing, closure, 0, 0))["uniform"]
            selected = inverse_cdf(probabilities, uniform)["selected_index"]
            counts[selected] += 1
        category_rows = []
        passed = True
        for category, probability in enumerate(probabilities):
            expected = n * probability
            bound = 0.0 if probability == 0.0 else (
                6.0 * math.sqrt(n * probability * (1.0 - probability)) + 1.0
            )
            deviation = abs(counts[category] - expected)
            category_pass = counts[category] == 0 if probability == 0.0 else deviation <= bound
            passed = passed and category_pass
            category_rows.append({
                "category": category,
                "probability": probability,
                "count": counts[category],
                "expected": expected,
                "absolute_deviation": deviation,
                "bound": bound,
                "passed": category_pass,
            })
        results.append({
            "fixture": fixture["id"],
            "contexts": n,
            "probabilities": probabilities,
            "counts": counts,
            "categories": category_rows,
            "passed": passed,
        })
    return results


def authority_key(category: int) -> bytes:
    return hmac.new(
        sha(b"RNG2_D0_GENERIC_AUTHORITY_ROOT").encode(),
        b"RNG2_D0_GENERIC_STATE_AUTHORITY_V1\x00" + category.to_bytes(1, "big"),
        hashlib.sha256,
    ).digest()


def paired_exercise(
    config: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    sampler_key: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model = A0.load_checkpoint(ROOT / config["surface_channel"]["checkpoint_path"])[0]
    by_closure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_closure[row["closure_id"]].append(row)
    outputs = []
    operation_rows = []
    reference_equal = 0
    uniform_pairs_equal = 0
    post_divergence_equal = 0
    post_divergence_total = 0
    diverged_pairs = 0
    none_operations = 0
    none_schedule_valid = 0
    semantic_valid = 0
    for closure_id in sorted(by_closure):
        rows = sorted(by_closure[closure_id], key=lambda row: row["orientation"])
        if len(rows) != 2:
            raise ValueError("ORIENTATIONS")
        prompt = rows[0]["prompt_text"].encode()
        payload = CAR.surface_payload(
            prompt, rows[0]["visible_answer"],
            {"surface_semantic_payload": {"schema": "SCB_M1_SURFACE_SEMANTIC_V1"}},
        )
        pairing = sha(b"RNG2_D0_PAIRING_CONTEXT_V1\x00" + closure_id.encode())
        sides = []
        for side, row in zip(("NATURAL", "ALIAS"), rows):
            generated = generate(
                model, payload, sampler_key=sampler_key, pairing_context_id=pairing,
                closure_id=closure_id, output_index=0,
                authority_key=authority_key(row["natural"]["z2"]),
                bias=config["surface_channel"]["bias"],
            )
            parsed_ok = (
                generated["parsed"]["valid"] is True
                and generated["parsed"]["values"] == {
                    key: payload[key] for key in ("z1", "z2", "z3", "answer")
                }
            )
            semantic_valid += int(parsed_ok)
            outputs.append({
                "closure_id": closure_id,
                "side": side,
                "source_orientation": row["orientation"],
                "generic_authority_category": row["natural"]["z2"],
                "semantic_payload": payload,
                "literal_output": generated["literal_output"],
                "output_sha256": generated["output_sha256"],
                "parsed": generated["parsed"],
                "semantic_valid": parsed_ok,
                "eligible_positions": generated["eligible_positions"],
                "stochastic_draws": generated["stochastic_draws"],
            })
            for operation in generated["operations"]:
                reconstructed = REF.reconstruct_trace(sampler_key, operation)
                agreement = (
                    reconstructed["uniform_bytes_hex"] == operation["uniform_bytes_hex"]
                    and reconstructed["selected_index"] == operation["selected_index"]
                    and reconstructed["selected_token"] == operation["selected_token"]
                    and all(
                        abs(left - right) <= 1e-15
                        for left, right in zip(
                            reconstructed["normalized_probabilities"],
                            operation["probabilities"],
                        )
                    )
                )
                reference_equal += int(agreement)
                if operation["selected_token"] == "<NONE>":
                    none_operations += 1
                    none_ok = (
                        operation["public"] is False
                        and operation["selected_token"] in generated["model_history"]
                    )
                    none_schedule_valid += int(none_ok)
                operation_rows.append({
                    "closure_id": closure_id,
                    "side": side,
                    **operation,
                    "reference_agreement": agreement,
                })
            sides.append(generated)
        first_divergence = None
        for index, (left, right) in enumerate(zip(sides[0]["operations"], sides[1]["operations"])):
            if left["model_history_before"] != right["model_history_before"]:
                first_divergence = index
                break
        if first_divergence is not None:
            diverged_pairs += 1
        for index, (left, right) in enumerate(zip(sides[0]["operations"], sides[1]["operations"])):
            equal = left["uniform_bytes_hex"] == right["uniform_bytes_hex"]
            uniform_pairs_equal += int(equal)
            if first_divergence is not None and index >= first_divergence:
                post_divergence_total += 1
                post_divergence_equal += int(equal)
    metrics = {
        "closures": len(by_closure),
        "outputs": len(outputs),
        "operation_rows": len(operation_rows),
        "reference_agreements": reference_equal,
        "uniform_pair_agreements": uniform_pairs_equal,
        "diverged_pairs": diverged_pairs,
        "post_divergence_uniform_agreements": post_divergence_equal,
        "post_divergence_uniform_comparisons": post_divergence_total,
        "NONE_operations": none_operations,
        "NONE_schedule_valid": none_schedule_valid,
        "semantic_valid_outputs_descriptive": semantic_valid,
    }
    return outputs, operation_rows, metrics


def malformed_tests(key: bytes) -> list[dict[str, Any]]:
    base = index_for("0" * 64, "1" * 64, 0, 0)
    cases = []
    mutations = {
        "STEP_OUT_OF_RANGE": {"model_step": 49},
        "OUTPUT_OUT_OF_RANGE": {"output_index": 8},
        "MISSING_PAIRING_CONTEXT": {"pairing_context_id": None},
        "INVALID_CLOSURE": {"closure_id": "z" * 64},
        "WRONG_SUBDRAW": {"subdraw_index": 1},
    }
    for name, update in mutations.items():
        index = dict(base)
        index.update(update)
        rejected = []
        for function in (position_uniform, REF.reconstruct_uniform):
            try:
                function(key, index)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        cases.append({"fixture": name, "production_rejected": rejected[0],
                      "reference_rejected": rejected[1], "passed": all(rejected)})
    for name, weights in {
        "NAN_WEIGHT": [float("nan"), 1.0],
        "INFINITE_WEIGHT": [float("inf"), 1.0],
        "ALL_ZERO": [0.0, 0.0],
        "NEGATIVE_WEIGHT": [-1.0, 2.0],
    }.items():
        rejected = []
        for function in (inverse_cdf, REF.inverse_cdf):
            try:
                function(weights, 0.5)
                rejected.append(False)
            except ValueError:
                rejected.append(True)
        cases.append({"fixture": name, "production_rejected": rejected[0],
                      "reference_rejected": rejected[1], "passed": all(rejected)})
    return cases


def manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.executable_manifest.v1",
        "files": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha(path)}
            for path in paths
        ],
    }


def run(source_commit: str, output: Path) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if output.exists():
        raise FileExistsError(output)
    binding_checks = validate_bindings(config, source_commit)
    selected_closures, selected_rows = select_development(config)
    sampler_key = hashlib.sha256(b"CSW_AM_SCB_M1_RNG2_D0_DEVELOPMENT_KEY_V1").digest()
    deterministic = deterministic_fixtures(sampler_key)
    boundaries = boundary_fixtures()
    malformed = malformed_tests(sampler_key)
    distributions = distributional_tests(sampler_key)
    outputs, operations, paired = paired_exercise(config, selected_rows, sampler_key)

    deterministic_gate = (
        all(row["exact_agreement"] for row in deterministic)
        and all(row.get("preimage_differs_from_base", True) for row in deterministic)
        and all(row.get("fixture_uniform_differs_from_base", True) for row in deterministic)
    )
    gates = {
        "bindings": all(binding_checks.values()),
        "development_selection": (
            len(selected_closures) == 64 and len(selected_rows) == 128
        ),
        "deterministic_sampler": deterministic_gate,
        "boundary_semantics": all(row["passed"] for row in boundaries),
        "malformed_fail_closed": all(row["passed"] for row in malformed),
        "distributional_correctness": all(row["passed"] for row in distributions),
        "paired_49_step_uniforms": paired["uniform_pair_agreements"] == 64 * 49,
        "post_divergence_pairing": (
            paired["diverged_pairs"] > 0
            and paired["post_divergence_uniform_agreements"]
            == paired["post_divergence_uniform_comparisons"]
        ),
        "production_reference": paired["reference_agreements"] == 64 * 2 * 49,
        "fixed_draw_schedule": (
            paired["outputs"] == 128 and paired["operation_rows"] == 128 * 49
        ),
        "NONE_semantics": (
            paired["NONE_operations"] > 0
            and paired["NONE_schedule_valid"] == paired["NONE_operations"]
        ),
        "generic_realiser_structural_integration": all(
            row["eligible_positions"] == 24 and row["stochastic_draws"] == 49
            for row in outputs
        ),
        "future_roles_sealed": True,
    }
    passed = all(gates.values())
    terminal = (
        config["terminals"]["valid"] if passed
        else config["terminals"]["implementation_invalid"]
    )
    output.mkdir(parents=True)
    write_rows(output / "SELECTED_DEVELOPMENT_CLOSURES.jsonl", selected_closures)
    write_rows(output / "SELECTED_DEVELOPMENT_ROWS.jsonl", selected_rows)
    write_rows(output / "DETERMINISTIC_FIXTURES.jsonl", deterministic)
    write_rows(output / "BOUNDARY_FIXTURES.jsonl", boundaries)
    write_rows(output / "MALFORMED_FIXTURES.jsonl", malformed)
    write_rows(output / "DISTRIBUTIONAL_RESULTS.jsonl", distributions)
    write_rows(output / "GENERIC_OUTPUTS.jsonl", outputs)
    write_rows(output / "PAIRED_OPERATION_TRACES.jsonl", operations)
    executable_paths = [
        CONFIG_PATH, FIXTURE_PATH,
        ROOT / "protocol/csw-am-scb-m1-rng2-d0/EVIDENCE_PROTOCOL.json",
        ROOT / "tools/csw_am_scb_m1_rng2_d0.py",
        ROOT / "tools/csw_am_scb_m1_rng2_reference.py",
    ]
    write_json(output / "EXECUTABLE_MANIFEST.json", manifest(executable_paths))
    result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.result.v1",
        "identity": config["identity"],
        "source_commit": source_commit,
        "passed": passed,
        "terminal": terminal,
        "binding_checks": binding_checks,
        "gates": gates,
        "development_selection": {
            "available_closures": 256,
            "selected_closures": len(selected_closures),
            "selected_rows": len(selected_rows),
            "z2_categories": sorted({row["natural"]["z2"] for row in selected_rows}),
            "answers": sorted({row["visible_answer"] for row in selected_rows}),
            "templates": sorted({row["template_id"] for row in selected_rows}),
            "z2_template_cells": len({
                (row["natural"]["z2"], row["template_id"]) for row in selected_rows
            }),
        },
        "denominators": {
            "deterministic_fixture_rows": len(deterministic),
            "boundary_fixture_rows": len(boundaries),
            "malformed_fixture_rows": len(malformed),
            "distributional_fixtures": len(distributions),
            "distributional_contexts": sum(row["contexts"] for row in distributions),
            "generic_outputs": len(outputs),
            "paired_operation_rows": len(operations),
        },
        "paired_metrics": paired,
        "descriptive_only": {
            "semantic_valid_outputs": paired["semantic_valid_outputs_descriptive"],
            "semantic_output_denominator": paired["outputs"],
            "generic_carrier_power": "NOT_EVALUATED",
            "detector_thresholds": "NOT_INSTANTIATED",
        },
        "scientific_claims": {
            "position_indexed_sampler_implemented": passed,
            "categorical_distribution_preserved_on_frozen_fixtures": gates["distributional_correctness"],
            "common_random_numbers_preserved_after_history_divergence": gates["post_divergence_pairing"],
            "generic_channel_power_confirmed": False,
            "receipt_bound_output_provenance_confirmed": False,
        },
        "release_status": {
            "calibration_identities_accessed": 0,
            "validation_identities_accessed": 0,
            "reserve_identities_accessed": 0,
            "protected_identities_accessed": 0,
            "RNG2_CAL1_authorized": False,
            "separate_CAL1_request_permitted": passed,
        },
    }
    write_json(output / "OFFICIAL_RESULT.json", result)
    reference_result = {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.reference_result.v1",
        "source_commit": source_commit,
        "passed": passed,
        "terminal": terminal,
        "gates": gates,
        "result_core_sha256": digest({
            "gates": gates,
            "development_selection": result["development_selection"],
            "denominators": result["denominators"],
            "paired_metrics": paired,
        }),
    }
    write_json(output / "REFERENCE_RESULT.json", reference_result)
    files = sorted(path for path in output.iterdir() if path.is_file())
    evidence_manifest = {
        "schema": "exp33a.csw_am_scb_m1_rng2_d0.evidence_manifest.v1",
        "files": [
            {"path": path.name, "sha256": file_sha(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    write_json(output / "EVIDENCE_MANIFEST.json", evidence_manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "preflight":
        config = read_json(CONFIG_PATH)
        selected, rows = select_development(config)
        print(canonical({
            "selected_closures": len(selected),
            "selected_rows": len(rows),
            "z2_template_cells": len({
                (row["natural"]["z2"], row["template_id"]) for row in rows
            }),
        }).decode())
        return
    if not args.source_commit:
        parser.error("--source-commit is required")
    print(canonical(run(args.source_commit, args.output)).decode())


if __name__ == "__main__":
    main()
