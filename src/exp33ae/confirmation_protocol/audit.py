"""Repository-authored, explicitly non-independent protocol-freeze audit."""

from __future__ import annotations

from pathlib import Path
import json

from ..remediation.strict import sha256_file
from .fixture import build
from .guard import assert_confirmation_disabled_in_candidate
from .normative import ACCEPTED_PREDECESSOR_TAG,ACCEPTED_PREDECESSOR_TAG_OBJECT,ACCEPTED_PREDECESSOR_COMMIT,ACCEPTED_PREDECESSOR_TREE,candidate_counters,git,load_protocol,offline_dependency_preflight,protocol_hash


def run(root:Path,full_fixture:bool=True)->dict:
    checks=[]
    def check(name,condition,detail):checks.append({"name":name,"passed":bool(condition),"detail":detail})
    protocol=load_protocol();preflight=offline_dependency_preflight();assert_confirmation_disabled_in_candidate()
    check("accepted predecessor tag object",git(root,"rev-parse",ACCEPTED_PREDECESSOR_TAG)==ACCEPTED_PREDECESSOR_TAG_OBJECT,git(root,"rev-parse",ACCEPTED_PREDECESSOR_TAG))
    check("accepted predecessor commit",git(root,"rev-parse",f"{ACCEPTED_PREDECESSOR_TAG}^{{commit}}")==ACCEPTED_PREDECESSOR_COMMIT,git(root,"rev-parse",f"{ACCEPTED_PREDECESSOR_TAG}^{{commit}}"))
    check("accepted predecessor tree",git(root,"rev-parse",f"{ACCEPTED_PREDECESSOR_TAG}^{{tree}}")==ACCEPTED_PREDECESSOR_TREE,git(root,"rev-parse",f"{ACCEPTED_PREDECESSOR_TAG}^{{tree}}"))
    check("offline dependency contract",preflight["passed"],preflight)
    check("candidate counters all zero",set(candidate_counters().values())=={0},candidate_counters())
    check("obsolete control gap absent",protocol["thresholds_integer"]["each_individual_control_gap"]==[48,160],protocol["thresholds_integer"]["each_individual_control_gap"])
    check("sole primary IIT",protocol["representation"]["primary_method"]=="fixed declared IIT alignment" and not protocol["representation"]["secondary_methods"],protocol["representation"])
    check("five R3 models",protocol["model_and_training"]["population"]==5 and protocol["model_and_training"]["recipe_id"]=="R3_MORE_STEPS_AND_WIDTH",protocol["model_and_training"])
    check("exact registry",protocol["registry"]["total"]==19600 and protocol["registry"]["total_per_model"]==3920,protocol["registry"])
    enabling = [path for path in root.rglob("*.json") if "schemas" not in path.parts and path.name in {"external-protocol-seal.json", "external-confirmation-authorization.json"}]
    check("confirmation enabling objects absent",not enabling,[str(path.relative_to(root)) for path in enabling])
    fixture=None
    if full_fixture:
        fixture=build(root);check("DEV_ONLY integration fixture",fixture["registry_rows"]==19600 and fixture["programme_count"]==702 and fixture["pair_count"]==400 and fixture["confirmation_objects_created"]==0,fixture["fixture_manifest_sha256"])
    return {"schema":"exp33ae.non_independent_protocol_freeze_audit.v1","independent":False,"protocol_sha256":protocol_hash(),"checks":checks,"passed_checks":sum(row["passed"] for row in checks),"total_checks":len(checks),"passed":all(row["passed"] for row in checks),"fixture_manifest_sha256":None if fixture is None else fixture["fixture_manifest_sha256"],"confirmation_objects_created":0,"confirmation_objects_accessed":0}
