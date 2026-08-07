"""Canonical persistence, payload indexing, and independent byte reopening."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from ..remediation.strict import canonical_json,object_sha256,sha256_file,strict_load_json,strict_load_jsonl,write_json,write_jsonl,StrictValidationError
from .allocation import AllocatedExample
from .normative import protocol_hash


INDEX_EXCLUSIONS={"access/access.lock","access/state.json","access/journal.jsonl","access/buffer.jsonl","scientific-index.json","derived-status.json","release-envelope.json","runtime-observations.json"}


def write_allocation(output:Path,allocation:dict,pairs:list[dict])->dict:
    directory=output/"allocation";write_jsonl(directory/"programmes.jsonl",allocation["programmes"]);write_jsonl(directory/"examples.jsonl",[row.as_dict() for row in allocation["examples"]]);write_jsonl(directory/"pairs.jsonl",pairs)
    manifest={"schema":"exp33ae.allocation_manifest.v1","namespace":allocation["namespace"],"stage":allocation["stage"],"claim_bearing":allocation["claim_bearing"],"protocol_sha256":protocol_hash(),"programme_count":702,"example_count":11232,"pair_count":400,"balance":allocation["balance"],"development_exclusion_manifest":allocation["development_exclusion_manifest"],
        "files":{name:sha256_file(directory/name) for name in ("programmes.jsonl","examples.jsonl","pairs.jsonl")}}
    manifest["allocation_sha256"]=object_sha256(manifest);write_json(directory/"manifest.json",manifest);return manifest


def load_allocation(output:Path)->tuple[dict,list[dict],dict]:
    directory=output/"allocation";manifest=strict_load_json(directory/"manifest.json");copy=dict(manifest);digest=copy.pop("allocation_sha256",None)
    if digest!=object_sha256(copy):raise StrictValidationError("ALLOCATION_BINDING","manifest")
    for name,file_hash in manifest["files"].items():
        if sha256_file(directory/name)!=file_hash:raise StrictValidationError("ALLOCATION_BINDING",name)
    programmes=strict_load_jsonl(directory/"programmes.jsonl");raw_examples=strict_load_jsonl(directory/"examples.jsonl");pairs=strict_load_jsonl(directory/"pairs.jsonl")
    examples=[]
    for row in raw_examples:
        value=dict(row);value["operation_ids"]=tuple(value["operation_ids"]);examples.append(AllocatedExample(**value))
    if len(programmes)!=702 or len(examples)!=11232 or len(pairs)!=400:raise StrictValidationError("ALLOCATION_BINDING","counts")
    return {"namespace":manifest["namespace"],"stage":manifest["stage"],"claim_bearing":manifest["claim_bearing"],"programmes":programmes,"examples":examples,"balance":manifest["balance"],"development_exclusion_manifest":manifest["development_exclusion_manifest"]},pairs,manifest


def write_registry(output:Path,registry:dict)->None:
    directory=output/"registry";write_jsonl(directory/"registry.jsonl",registry["rows"]);manifest=dict(registry["manifest"]);manifest["registry_file_sha256"]=sha256_file(directory/"registry.jsonl");manifest["registry_manifest_sha256"]=object_sha256({k:v for k,v in manifest.items() if k!="registry_manifest_sha256"});write_json(directory/"manifest.json",manifest)


def build_scientific_index(output:Path)->dict:
    files=[]
    for path in sorted(output.rglob("*")):
        if not path.is_file():continue
        relative=str(path.relative_to(output))
        if relative in INDEX_EXCLUSIONS or "__pycache__" in path.parts:continue
        files.append({"path":relative,"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    value={"schema":"exp33ae.scientific_payload_index.v1","protocol_sha256":protocol_hash(),"exclusions":sorted(INDEX_EXCLUSIONS),"files":files,"file_count":len(files),"payload_sha256":sha256(canonical_json(files).encode()).hexdigest()}
    return value


def freeze_scientific_index(output:Path)->dict:
    value=build_scientific_index(output);write_json(output/"scientific-index.json",value);return value


def verify_scientific_index(output:Path)->dict:
    stored=strict_load_json(output/"scientific-index.json");observed=build_scientific_index(output)
    return {"matches":stored==observed,"stored":stored,"observed":observed}
