"""Conspicuous confirmation-ineligible DEV_ONLY integration fixture."""

from __future__ import annotations

from pathlib import Path

from ..remediation.strict import object_sha256,write_json
from .allocation import allocate,build_pairs
from .entropy import dev_only_golden_context,golden_vectors
from .normative import DEV_NAMESPACE,DEV_STAGE,candidate_counters,protocol_hash
from .registry import build_registry,fixture_model_bindings
from .representation import representation_manifest
from .status import qualifying_fixture,derive_status


def build(root:Path,write_to:Path|None=None)->dict:
    entropy=dev_only_golden_context();allocation=allocate(root,entropy);pairs=build_pairs(allocation,entropy);representation=representation_manifest(DEV_NAMESPACE,DEV_STAGE,False);training=fixture_model_bindings(entropy);registry=build_registry(allocation,pairs,entropy,representation,training,fixture_norm=1.0)
    seeds=list(entropy.model_seeds());metrics=[qualifying_fixture(order,seed) for order,seed in enumerate(seeds)];decision=derive_status(metrics,seeds)
    value={"schema":"exp33ae.dev_only_protocol_freeze_fixture_manifest.v1","namespace":DEV_NAMESPACE,"stage":DEV_STAGE,"claim_bearing":False,"confirmation_eligible":False,"protocol_sha256":protocol_hash(),
        "entropy_golden":golden_vectors(),"programme_count":len(allocation["programmes"]),"example_count":len(allocation["examples"]),"pair_count":len(pairs),"registry_rows":len(registry["rows"]),
        "registry_payload_commitment":registry["manifest"]["payload_commitment"],"registry_manifest_sha256":registry["manifest"]["registry_manifest_sha256"],"ordered_model_seed_identities":seeds,
        "synthetic_threshold_branch":"DEV_ONLY_SIMULATED_ALL_GATES_PASS" if decision["status"].endswith("POSITIVE") else "DEV_ONLY_SIMULATED_BRANCH_FAILURE",
        "terminal_confirmation_status_materialized":False,"synthetic_threshold_evidence_is_trained_model_evidence":False,"candidate_counters":candidate_counters(),"confirmation_objects_created":0,"confirmation_objects_accessed":0}
    value["fixture_manifest_sha256"]=object_sha256(value)
    if write_to is not None:write_json(write_to,value)
    return value
