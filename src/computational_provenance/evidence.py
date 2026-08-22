"""Independent integrity checks for the four claim-bearing end-to-end evaluations."""
from __future__ import annotations

import gzip
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _plain_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as stream:
            return stream.read()
    return path.read_bytes()


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _derived_key(master: bytes, purpose: str) -> bytes:
    return hmac.new(master, canonical(["CSW-AM-SCB-M1-REC", purpose, "V1"]), hashlib.sha256).digest()


def _verify_receipts(directory: Path, protocol: dict[str, Any], master: bytes) -> tuple[int, int]:
    packages = envelopes = 0
    for package in _rows(directory / "RECEIPT_PACKAGES.jsonl.gz"):
        packages += 1
        if set(package["receipts"]) != {"exact_z2", "abstract_z2", "exact_z3", "abstract_z3", "chain"}:
            raise ValueError(f"receipt set: {directory}")
        for envelope in package["receipts"].values():
            receipt_type, preimage = envelope["receipt_type"], envelope["preimage"]
            domain = protocol["receipt_types"][receipt_type]["domain"]
            expected_id = _sha(canonical([domain, preimage]))
            expected_tag = hmac.new(_derived_key(master, receipt_type), canonical(preimage), hashlib.sha256).hexdigest()
            if envelope["receipt_id"] != expected_id or not hmac.compare_digest(envelope["tag"], expected_tag):
                raise ValueError(f"invalid receipt: {directory.name}/{package['package_id']}")
            envelopes += 1
    return packages, envelopes


def verify_evaluation(architecture: str, split: str) -> dict[str, Any]:
    directory = ROOT / f"evidence/full/{architecture}/{split}"
    raw_commitment = json.loads((directory / "RAW_EVIDENCE_COMMITMENT.json").read_text())
    main_audit = json.loads((directory / "MAIN_AUDIT.json").read_text())
    terminal = json.loads((directory / "TERMINAL.json").read_text())
    reveal = json.loads((directory / "KEY_REVEAL.json").read_text())

    raw_hashes = {}
    for name, expected in raw_commitment["files"].items():
        path = directory / (name + ".gz")
        observed = _sha(_plain_bytes(path))
        if observed != expected:
            raise ValueError(f"raw commitment mismatch: {path}")
        raw_hashes[name] = observed

    production_files = {
        "aggregates_sha256": "AGGREGATE_SCORES.jsonl.gz",
        "attacks_sha256": "ATTACK_VERDICTS.jsonl.gz",
        "closures_sha256": "CLOSURE_DECISIONS.jsonl.gz",
        "genuine_sha256": "GENUINE_VERDICTS.jsonl.gz",
        "primitive_sha256": "PRIMITIVE_SCORES.jsonl.gz",
        "official_sha256": "OFFICIAL_RESULT.json",
        "raw_commitment_sha256": "RAW_EVIDENCE_COMMITMENT.json",
        "reference_sha256": "REFERENCE_RESULT.json",
    }
    for key, name in production_files.items():
        observed = _sha(_plain_bytes(directory / name))
        if observed != main_audit["evidence"][key]:
            raise ValueError(f"adjudication commitment mismatch: {directory / name}")
    if _sha((directory / "MAIN_AUDIT.json").read_bytes()) != terminal["main_audit_sha256"]:
        raise ValueError(f"terminal commitment mismatch: {directory}")
    raw_commitment_hash = _sha((directory / "RAW_EVIDENCE_COMMITMENT.json").read_bytes())
    if reveal["raw_commitment_sha256"] != raw_commitment_hash:
        raise ValueError(f"key chronology mismatch: {directory}")

    receipt_protocol = json.loads((ROOT / f"protocols/{architecture}_public_receipts.json").read_text())
    packages, envelopes = _verify_receipts(directory, receipt_protocol, bytes.fromhex(reveal["receipt_key_hex"]))

    output_count = model_steps = eligible_positions = 0
    condition_counts: dict[str, int] = {}
    for row in _rows(directory / "OUTPUTS.jsonl.gz"):
        output_count += 1
        model_steps += row["model_steps"]
        eligible_positions += (len(row["draws"]) - 1) // 2  # 49 indexed steps contain 24 eligible carrier positions.
        condition_counts[row["condition"]] = condition_counts.get(row["condition"], 0) + 1
        if row["model_steps"] != 49 or len(row["draws"]) != 49:
            raise ValueError(f"non-frozen generation length: {directory}")
        if _sha(row["literal_output"].encode()) != row["literal_output_sha256"]:
            raise ValueError(f"literal output digest: {directory}")
    counts = {
        "execution_packages": sum(1 for _ in _rows(directory / "EXECUTION_ROWS.jsonl.gz")),
        "receipt_packages": packages,
        "receipt_envelopes": envelopes,
        "generation_authorizations": sum(1 for _ in _rows(directory / "GENERATION_AUTHORIZATIONS.jsonl.gz")),
        "outputs": output_count,
        "model_steps": model_steps,
        "eligible_positions": eligible_positions,
        "attacks": sum(1 for _ in _rows(directory / "ATTACK_INPUTS.jsonl.gz")),
        "closures": sum(1 for _ in _rows(directory / "CLOSURE_DECISIONS.jsonl.gz")),
    }
    expected = {"execution_packages": 256, "receipt_packages": 256, "receipt_envelopes": 1280,
                "generation_authorizations": 1280, "outputs": 10240, "model_steps": 501760,
                "eligible_positions": 245760, "attacks": 1024, "closures": 128}
    if counts != expected or condition_counts != {name: 2048 for name in ("STATE", "CARRIER_OFF", "PROMPT", "ANSWER", "GENERIC")}:
        raise ValueError(f"claim-bearing denominator mismatch: {directory}: {counts} {condition_counts}")
    official = json.loads((directory / "OFFICIAL_RESULT.json").read_text())
    reference = json.loads((directory / "REFERENCE_RESULT.json").read_text())
    if split == "public":
        passed = bool(main_audit["valid"] and official["passed"] and reference["passed"] and terminal["valid"])
    else:
        passed = bool(main_audit["scientific_positive"] and main_audit["evidence_valid"] and official["passed"]
                      and reference["scientific_positive"] and terminal["scientific_positive"] and terminal["evidence_valid"])
    return {"architecture": architecture, "split": split, "identity": main_audit["identity"], "counts": counts,
            "condition_counts": condition_counts, "all_receipts_verified": True, "all_commitments_verified": True,
            "terminal": terminal["terminal"], "passed": passed}


def verify_all_evidence() -> dict[str, Any]:
    evaluations = [verify_evaluation(architecture, split) for architecture in ("feedforward", "transformer") for split in ("public", "protected")]
    return {"evaluations": evaluations, "all_four_passed": all(row["passed"] for row in evaluations)}
