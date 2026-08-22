import copy
import hashlib

from computational_provenance.carrier import aggregate, detected, generate, surface_payload
from computational_provenance.receipts import issue_execution, verify_execution


def test_receipt_tampering_is_rejected() -> None:
    key = bytes(range(32))
    receipt = issue_execution(key, model_domain="test", checkpoint_sha256="a" * 64,
                              closure_id="closure", role="NATURAL", z2=2, z3=6, answer=6)
    assert verify_execution(key, receipt)
    tampered = copy.deepcopy(receipt)
    tampered["z2"]["payload"]["category"] = 10
    assert not verify_execution(key, tampered)


def test_paired_randomness_and_state_switch() -> None:
    key = hashlib.sha256(b"sampler").digest()
    payload = surface_payload("same prompt", 6)
    low = generate(payload, 2, key, "closure", 0)
    high = generate(payload, 10, key, "closure", 0)
    assert low["semantic_payload"] == high["semantic_payload"]
    assert [item["position"] for item in low["eligible"]] == list(range(24))
    assert [item["position"] for item in high["eligible"]] == list(range(24))
    assert low["text"] != high["text"]


def test_detector_recovers_embedded_category() -> None:
    key = hashlib.sha256(b"sampler-test").digest()
    payload = surface_payload("same prompt", 6)
    for category in (2, 10):
        outputs = [generate(payload, category, key, "closure", index) for index in range(8)]
        scores = aggregate(outputs)
        assert detected(scores, category)
