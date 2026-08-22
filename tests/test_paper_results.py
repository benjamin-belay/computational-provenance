from computational_provenance.answer_only import reproduce_competence, reproduce_localisation
from computational_provenance.evidence import verify_all_evidence
from computational_provenance.robustness import reproduce_robustness


def test_all_final_evidence_and_receipts_verify():
    result = verify_all_evidence()
    assert result["all_four_passed"]
    assert all(row["counts"]["outputs"] == 10240 for row in result["evaluations"])


def test_claim_bearing_robustness_denominators():
    feedforward = reproduce_robustness("feedforward")
    transformer = reproduce_robustness("transformer")
    assert feedforward["models_passed"] == 5
    assert transformer["models_passed"] == 3
    assert all(result["held_out_rows_per_model"] == 4096 for result in (feedforward, transformer))
    assert all(result["causal_cases_per_model"] == 28672 for result in (feedforward, transformer))


def test_answer_only_competence_and_localisation():
    competence = reproduce_competence()
    localisation = reproduce_localisation()
    assert competence["three_of_three_passed"]
    assert localisation["full_qualified_sites"] == 0
    assert localisation["quotient_qualified_sites"] == 0
    assert localisation["candidate_sites"] == 20
    assert localisation["all_activation_digests_match"]
    assert round(localisation["best_z2"]["accuracy"] * 100, 2) == 23.80
    assert round(localisation["best_quotient"]["accuracy"] * 100, 2) == 45.54
