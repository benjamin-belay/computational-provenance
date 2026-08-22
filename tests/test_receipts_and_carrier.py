import json

from computational_provenance.carrier import score_literal
from computational_provenance.evidence import ROOT, _rows
from computational_provenance.receipts import evaluate_bundle


def test_frozen_literal_is_parsed_and_scored() -> None:
    directory = ROOT / "evidence/full/feedforward/public"
    output = next(iter(_rows(directory / "OUTPUTS.jsonl.gz")))
    primitive = next(iter(_rows(directory / "PRIMITIVE_SCORES.jsonl.gz")))
    parsed, counts, normalized = score_literal(output["literal_output"])
    assert parsed["valid"]
    assert counts == primitive["green_counts"]
    assert normalized == primitive["normalized_scores"]


def test_frozen_receipt_bundle_is_independently_accepted() -> None:
    directory = ROOT / "evidence/full/feedforward/public"
    execution = next(iter(_rows(directory / "EXECUTION_ROWS.jsonl.gz")))
    package = next(iter(_rows(directory / "RECEIPT_PACKAGES.jsonl.gz")))
    protocol = json.loads((ROOT / "protocols/feedforward_public_receipts.json").read_text())
    key = bytes.fromhex(json.loads((directory / "KEY_REVEAL.json").read_text())["receipt_key_hex"])
    expected = {name: execution[name] for name in ("run_id", "context_id", "execution_id", "nonce")}
    expected["checkpoint_sha256"] = protocol["state_ontology"]["checkpoint_sha256"]
    assert all(value == "ACCEPT" for value in evaluate_bundle(key, package["receipts"], expected, protocol).values())
