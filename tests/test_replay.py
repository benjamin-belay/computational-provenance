from computational_provenance.pipeline import replay


def test_four_case_feedforward_replay() -> None:
    result = replay("feedforward", "public", limit=4)
    assert result["closures"] == 4
    assert result["complete_successes"] == 4


def test_four_case_transformer_replay() -> None:
    result = replay("transformer", "public", limit=4)
    assert result["closures"] == 4
    assert result["complete_successes"] == 4
