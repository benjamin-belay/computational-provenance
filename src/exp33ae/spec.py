"""Prospectively fixed DEV_ONLY task constants and token schema."""

from __future__ import annotations

from hashlib import sha256

NAMESPACE = "DEV_ONLY"
K = 2
PARTITIONS = (
    "training",
    "model_selection",
    "localization_fit",
    "localization_selection",
    "causal_evaluation",
    "stress_integrity",
)

PROGRAM_COUNTS = {
    "training": 512,
    "model_selection": 30,
    "localization_fit": 30,
    "localization_selection": 30,
    "causal_target_composition": 50,
    "causal_parallel_composition": 50,
    "stress_integrity": 19,
}

MODEL_SEEDS = {
    "ordinary": (1012, 2023, 3034),
    "iit_primary": (1112, 2223, 3334),
    "undertrained": (4045,),
}

# Four balanced Boolean operations, presented by their complete transition tables.
# Evaluation holds out compositions of known operations, not unlearnable operation
# identities. Every programme identity is the six actual transition functions.
ALL_OPERATIONS = (
    (1, 1, 0, 0),
    (1, 0, 1, 0),
    (1, 0, 0, 1),
    (0, 1, 1, 0),
)
TRAIN_OPERATION_IDS = (0, 1, 2, 3)
HELDOUT_OPERATION_IDS = ()
INTEGRITY_OPERATION_IDS = ()

# Token sequence plus structured operation-table features.
BOS = 0
ROLE_TOKENS = {"t1": 1, "t2": 2, "t3": 3, "n1": 4, "n2": 5, "n3": 6}
INPUT_MARKER = 7
QUERY = 8
Z1_SLOT = 9
N1_SLOT = 10
Z2_SLOT = 11
N2_SLOT = 12
NUMBER_OFFSET = 13  # shared input-bit tokens 0 and 1
VOCAB_SIZE = 15
SEQ_LEN = 17
ROLE_POSITIONS = (1, 2, 3, 4, 5, 6)
Z1_POSITION = 12
N1_POSITION = 13
Z2_POSITION = 14
N2_POSITION = 15
QUERY_POSITION = 16
JOINT_CLASSES = K * K

# The primary IIT declaration is fixed before transformer training.
IIT_LAYER = 2
IIT_POSITION = Z2_POSITION
IIT_RANK = K
IIT_STRENGTH = 1.0

# Final DEV_ONLY scientific gates, declared before model training/evaluation.
GATES = {
    "task_accuracy_each_axis": 0.85,
    "minimum_programme_accuracy": 0.75,
    "shortcut_advantage": 0.20,
    "decode_accuracy": 0.90,
    "different_state_iia": 0.85,
    "state_transport": 0.90,
    "descendant_correctness": 0.85,
    "non_descendant_invariance": 0.95,
    "same_state_preservation": 0.95,
    "same_answer_different_state_iia": 0.80,
    "random_direction_gap": 0.30,
    "orthogonal_direction_gap": 0.30,
    "random_location_gap": 0.30,
    # Obsolete legacy name retained only for compatibility. The only admissible
    # layer control is layer-0 alternate_early_layer and its gap is 48/160.
    "random_layer_gap": 0.30,
    "call_base_source_accuracy": 0.85,
    "call_completion": 1.0,
    "required_primary_seeds": 3,
}

SEARCH_GRID = {
    "methods": ("linear_probe", "das"),
    "layers": (0, 1, 2, 3),
    "positions": (Z2_POSITION, QUERY_POSITION),
    "ranks": (1, 2),
    "strengths": (0.75, 1.0),
}

CALL_COUNTS_PER_SEED = {
    "causal_different_state": 160,
    "causal_same_state": 80,
    "causal_same_answer_different_state": 80,
    "matched_random_direction": 160,
    "matched_orthogonal_direction": 160,
    "matched_random_location": 160,
    "matched_random_layer": 160,
}


def stable_digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()
