"""Executable normative future training recipe without running confirmation training."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256

import torch

from ..model import ModelConfig, TinyProgrammeTransformer
from ..spec import (
    BOS, INPUT_MARKER, N1_SLOT, N2_SLOT, NUMBER_OFFSET, QUERY, ROLE_TOKENS,
    SEQ_LEN, VOCAB_SIZE, Z1_SLOT, Z2_SLOT,
)
from ..training import TrainConfig
from .entropy import DevEntropy
from .normative import DEV_NAMESPACE, load_protocol


def normative_training_table() -> dict:
    return load_protocol()["training"]


def validate_training_recipe_against_implementation() -> dict:
    table = normative_training_table()
    config = ModelConfig()
    model = TinyProgrammeTransformer(config)
    observed_tokens = {
        "BOS": BOS, "INPUT_MARKER": INPUT_MARKER, "N1_SLOT": N1_SLOT, "N2_SLOT": N2_SLOT,
        "NUMBER_OFFSET": NUMBER_OFFSET, "QUERY": QUERY, "ROLE_TOKENS": ROLE_TOKENS,
        "Z1_SLOT": Z1_SLOT, "Z2_SLOT": Z2_SLOT,
    }
    expected_architecture = {key: value for key, value in table["architecture"].items() if key not in {"parameter_count", "parameter_tolerance"}}
    observed_architecture = config.as_dict()
    default_train = asdict(TrainConfig())
    checks = {
        "architecture_exact": observed_architecture == expected_architecture,
        "parameter_count_exact": sum(parameter.numel() for parameter in model.parameters()) == table["architecture"]["parameter_count"],
        "tokenizer_exact": observed_tokens == table["tokenizer"],
        "sequence_length_exact": SEQ_LEN == 17 and VOCAB_SIZE == 15,
        "steps_exact": default_train["steps"] == table["steps"] == 1400,
        "batch_size_exact": default_train["batch_size"] == table["batch_size"] == 256,
        "optimizer_hyperparameters_exact": (
            default_train["learning_rate"] == table["optimizer"]["learning_rate"]
            and default_train["weight_decay"] == table["optimizer"]["weight_decay"]
        ),
        "loss_weights_exact": all(default_train[name] == 1.0 for name in ("iit_weight", "z2_code_weight", "n3_code_weight")),
        "registered_intervention_exact": table["iit"] == {"layer": 2, "position": 14, "rank": 2, "strength": 1.0},
        "final_step_only": table["checkpoint"].startswith("final step 1400 only"),
        "no_early_stopping": table["no_early_stopping"] is True,
        "no_seed_replacement": table["no_seed_replacement_or_retry"] is True,
    }
    return {
        "schema": "exp33ae.training_recipe_consistency.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "checks": checks, "passed": all(checks.values()),
        "observed_architecture": observed_architecture,
        "observed_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def dev_only_schedule_golden(entropy: DevEntropy, rows: int = 8192, preview_batches: int = 3) -> dict:
    if rows != 8192 or preview_batches != 3:
        raise ValueError("golden schedule shape is fixed")
    models = []
    for order, seed in enumerate(entropy.ordered_model_seeds()):
        batch_seed = int.from_bytes(entropy.derive("training-batch-stream", order, 8), "big") & ((1 << 63) - 1)
        interchange_seed = int.from_bytes(entropy.derive("training-interchange-stream", order, 8), "big") & ((1 << 63) - 1)
        batch_generator = torch.Generator().manual_seed(batch_seed)
        source_generator = torch.Generator().manual_seed(interchange_seed)
        batches, sources = [], []
        for _ in range(preview_batches):
            batches.append(torch.randint(0, rows, (256,), generator=batch_generator).tolist())
            sources.append(torch.randint(0, rows, (256,), generator=source_generator).tolist())
        payload = {"model_order": order, "model_seed_identity": seed, "batch_stream_seed": batch_seed, "interchange_stream_seed": interchange_seed, "base_index_batches": batches, "source_index_batches": sources}
        payload["schedule_preview_sha256"] = sha256(str(payload).encode()).hexdigest()
        models.append(payload)
    return {
        "schema": "exp33ae.dev_only_training_schedule_golden.v1", "namespace": DEV_NAMESPACE,
        "claim_bearing": False, "models": models, "full_steps": 1400,
        "pairs_per_step": 256, "full_interchange_pairs": 358400,
        "confirmation_training_started": False,
    }
