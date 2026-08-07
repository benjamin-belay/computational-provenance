"""Prospective catalogue, split, oracle, and four-stratum pair construction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path

from ..independent_oracle import oracle, oracle_do_z2
from ..spec import ALL_OPERATIONS
from ..remediation.strict import canonical_json, object_sha256, StrictValidationError
from .entropy import EntropyContext
from .normative import DEV_NAMESPACE, DEV_STAGE, load_protocol


EXCLUSION_SOURCES = (
    "registries/dev_only/programme-manifest.jsonl",
    "registries/dev_only/example-index.jsonl",
    "registries/dev_only/evaluation-call-registry.jsonl",
    "artifacts/dev_only/preconfirmation-revision-v1/programme-allocation.jsonl",
    "artifacts/dev_only/preconfirmation-revision-v1/example-allocation.jsonl",
    "artifacts/dev_only/preconfirmation-revision-v1/primary-pairs.jsonl",
    "artifacts/dev_only/preconfirmation-revision-v1/registry.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2-failed-at-registry-1/allocation/programmes.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2-failed-at-registry-1/allocation/examples.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2-failed-at-registry-1/allocation/pairs.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2/allocation/programmes.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2/allocation/examples.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2/allocation/pairs.jsonl",
    "artifacts/dev_only/33a-e-final-remediation-v2/registry/registry.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1-failed-before-training-1/allocation/programmes.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1-failed-before-training-1/allocation/examples.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1-failed-before-training-1/allocation/pairs.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1/allocation/programmes.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1/allocation/examples.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1/allocation/pairs.jsonl",
    "artifacts/dev_only/33a-e-training-reliability-v1/registry/registry.jsonl",
)

PARTITION_COUNTS = {
    "training": 512,
    "model_selection": 30,
    "localization_fit": 30,
    "localization_selection": 30,
    "clean_target_axis": 50,
    "clean_parallel_axis": 50,
}
STRATUM_COUNTS = {"A":160,"B":80,"C":80,"D":80}


def programme_id(operation_ids: tuple[int, ...]) -> str:
    payload = ";".join("".join(map(str, ALL_OPERATIONS[index])) for index in operation_ids)
    return "pgm-" + sha256(payload.encode()).hexdigest()[:16]


def composition_id(prefix: str, operation_ids: tuple[int, int, int]) -> str:
    payload = ";".join("".join(map(str, ALL_OPERATIONS[index])) for index in operation_ids)
    return prefix + sha256(payload.encode()).hexdigest()[:16]


def causal_signature(operation_ids: tuple[int, ...]) -> str:
    operations = tuple(ALL_OPERATIONS[index] for index in operation_ids)
    return sha256(canonical_json([oracle(operations, inputs) for inputs in product(range(2), repeat=4)]).encode()).hexdigest()


def _production_interpret(operations, inputs) -> dict:
    def step(table, left, right):
        return table[2 * left + right]
    a,b,c,d = inputs
    z1 = step(operations[0], a, b); z2 = step(operations[1], z1, c); z3 = step(operations[2], z2, d)
    n1 = step(operations[3], a, d); n2 = step(operations[4], n1, b); n3 = step(operations[5], n2, c)
    return {"z1":z1,"z2":z2,"z3":z3,"n1":n1,"n2":n2,"n3":n3,"joint_answer":2*z3+n3}


def is_balanced(operation_ids: tuple[int, ...]) -> bool:
    operations = tuple(ALL_OPERATIONS[index] for index in operation_ids)
    rows = [oracle(operations, inputs) for inputs in product(range(2), repeat=4)]
    return all(Counter(row[name] for row in rows) == Counter({0:8,1:8}) for name in ("z2","z3","n3")) and Counter(row["joint_answer"] for row in rows) == Counter({0:4,1:4,2:4,3:4})


@dataclass(frozen=True)
class AllocatedExample:
    namespace: str
    stage: str
    claim_bearing: bool
    partition: str
    axis: str
    programme_id: str
    causal_signature: str
    target_composition_id: str
    parallel_composition_id: str
    operation_ids: tuple[int, ...]
    a: int; b: int; c: int; d: int
    z1: int; z2: int; z3: int; n1: int; n2: int; n3: int; answer: int
    example_id: str

    @property
    def tokens(self):
        return (0,1,2,3,4,5,6,7,13+self.a,13+self.b,13+self.c,13+self.d,9,10,11,12,8)

    @property
    def operation_tables(self):
        return tuple(ALL_OPERATIONS[index] for index in self.operation_ids)

    @property
    def joint_answer(self):
        return self.answer

    def as_dict(self):
        result = dict(self.__dict__); result["operation_ids"] = list(self.operation_ids); return result


def development_exclusions(root: Path) -> dict:
    result = {name:set() for name in ("programme_ids","causal_signatures","example_ids","pair_ids","call_ids","nonces")}
    source_hashes = []
    for relative in EXCLUSION_SOURCES:
        path = root / relative
        if not path.is_file():
            raise StrictValidationError("DEVELOPMENT_EXCLUSION", f"missing frozen exclusion source {relative}")
        source_hashes.append({"path":relative,"sha256":sha256(path.read_bytes()).hexdigest()})
        for line in path.read_text().splitlines():
            if not line.strip():
                raise StrictValidationError("DEVELOPMENT_EXCLUSION", f"blank row {relative}")
            row = json.loads(line)
            for key, field in (("programme_ids","programme_id"),("causal_signatures","causal_signature"),("example_ids","example_id"),("pair_ids","pair_id"),("call_ids","call_id"),("nonces","nonce")):
                if isinstance(row.get(field), str): result[key].add(row[field])
    manifest = {
        "schema":"exp33ae.development_exclusion_manifest.v1",
        "sources":source_hashes,
        "counts":{key:len(value) for key,value in result.items()},
        "identity_sha256":object_sha256({key:sorted(value) for key,value in result.items()}),
    }
    return {**result,"manifest":manifest}


def _score(entropy: EntropyContext, domain: str, value) -> bytes:
    return sha256(entropy.derive(domain,0,32) + canonical_json(value).encode()).digest()


def _example(entropy: EntropyContext, partition: str, axis: str, ids: tuple[int,...], inputs) -> AllocatedExample:
    operations = tuple(ALL_OPERATIONS[index] for index in ids)
    independent = oracle(operations, inputs); production = _production_interpret(operations, inputs)
    if independent != production:
        raise StrictValidationError("ORACLE_DISAGREEMENT", canonical_json([ids,inputs]))
    prefix = "devonly-cpex-" if not entropy.claim_bearing else "exp33ae-cnfex-"
    example_id = prefix + sha256(canonical_json([entropy.stage,programme_id(ids),*inputs]).encode()).hexdigest()[:24]
    return AllocatedExample(entropy.namespace, entropy.stage, entropy.claim_bearing, partition, axis,
        programme_id(ids),causal_signature(ids),composition_id("tcomp-",ids[:3]),composition_id("ncomp-",ids[3:]),ids,
        *inputs, independent["z1"],independent["z2"],independent["z3"],independent["n1"],independent["n2"],independent["n3"],independent["joint_answer"],example_id)


def allocate(root: Path, entropy: EntropyContext) -> dict:
    exclusions = development_exclusions(root)
    triples = list(product(range(4), repeat=3))
    target = sorted(triples,key=lambda item:_score(entropy,"programme-allocation",["target",item]))
    parallel = sorted(triples,key=lambda item:_score(entropy,"programme-allocation",["parallel",item]))
    # Forty-eight revealed / sixteen held-out compositions is frozen for v1. It
    # leaves sufficient development-disjoint balanced programmes while keeping
    # a substantial within-run composition holdout on each axis.
    train_t,held_t = target[:48],target[48:]; train_n,held_n = parallel[:48],parallel[48:]
    specs = (
        ("training","revealed_training",train_t,train_n,512),
        ("model_selection","revealed_model_selection",train_t,train_n,30),
        ("localization_fit","revealed_localization_fit",train_t,train_n,30),
        ("localization_selection","revealed_localization_selection",train_t,train_n,30),
        ("clean_target_axis","blinded_within_run_target_composition",held_t,train_n,50),
        ("clean_parallel_axis","blinded_within_run_parallel_composition",train_t,held_n,50),
    )
    chosen={}; used=set()
    for partition,axis,tpool,npool,count in specs:
        candidates=[]
        for target_ids in tpool:
            for parallel_ids in npool:
                ids=(*target_ids,*parallel_ids); pid=programme_id(ids); sig=causal_signature(ids)
                if pid in exclusions["programme_ids"] or sig in exclusions["causal_signatures"] or pid in used or not is_balanced(ids): continue
                candidates.append(ids)
        candidates.sort(key=lambda item:_score(entropy,"split-allocation",[partition,item]))
        if len(candidates)<count: raise StrictValidationError("ALLOCATION_CONSTRAINT",f"{partition}: {len(candidates)} < {count}")
        chosen[partition]=tuple(candidates[:count]); used.update(programme_id(ids) for ids in chosen[partition])
    programmes=[]; examples=[]
    for partition,axis,*_ in specs:
        for ids in chosen[partition]:
            programmes.append({"namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
                "partition":partition,"axis":axis,"programme_id":programme_id(ids),"causal_signature":causal_signature(ids),
                "operation_ids":list(ids),"operations":[list(ALL_OPERATIONS[index]) for index in ids],
                "target_composition_id":composition_id("tcomp-",ids[:3]),"parallel_composition_id":composition_id("ncomp-",ids[3:])})
            examples.extend(_example(entropy,partition,axis,ids,inputs) for inputs in product(range(2),repeat=4))
    if len(programmes)!=702 or len(examples)!=11232 or len({row["programme_id"] for row in programmes})!=702 or len({row["causal_signature"] for row in programmes})!=702:
        raise StrictValidationError("ALLOCATION_CONSTRAINT","exact functional totals")
    if set(row.example_id for row in examples)&exclusions["example_ids"]:
        raise StrictValidationError("DEVELOPMENT_EXCLUSION","example overlap")
    balance={}
    for partition,count in PARTITION_COUNTS.items():
        rows=[row for row in examples if row.partition==partition]; expected=count*16
        if len(rows)!=expected: raise StrictValidationError("ALLOCATION_CONSTRAINT",partition)
        value={name:dict(sorted(Counter(getattr(row,name) for row in rows).items())) for name in ("z2","z3","n3","answer")}
        if value!={"z2":{0:expected//2,1:expected//2},"z3":{0:expected//2,1:expected//2},"n3":{0:expected//2,1:expected//2},"answer":{0:expected//4,1:expected//4,2:expected//4,3:expected//4}}:
            raise StrictValidationError("ALLOCATION_BALANCE",partition)
        balance[partition]=value
    result={"schema":"exp33ae.allocation.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
        "programmes":programmes,"examples":examples,"target_training_compositions":[composition_id("tcomp-",x) for x in train_t],
        "target_heldout_compositions":[composition_id("tcomp-",x) for x in held_t],"parallel_training_compositions":[composition_id("ncomp-",x) for x in train_n],
        "parallel_heldout_compositions":[composition_id("ncomp-",x) for x in held_n],"balance":balance,"development_exclusion_manifest":exclusions["manifest"]}
    return result


def _axis(row): return 0 if row.axis.endswith("target_composition") else 1
def _category(row): return (_axis(row),row.z2,row.z3,row.n3)
def _counterfactual(base,source_z2):
    operations=tuple(ALL_OPERATIONS[index] for index in base.operation_ids); z3=oracle_do_z2(operations,base.d,source_z2)["z3"]
    return {"z2":source_z2,"z3":z3,"n3":base.n3,"answer":2*z3+base.n3}
def _source_category(stratum,category):
    axis,z2,z3,n3=category
    if stratum=="A": return 1-axis,1-z2,1-z3,n3
    if stratum in {"B","D"}: return 1-axis,1-z2,z3,n3
    return 1-axis,z2,z3,n3


def build_pairs(allocation: dict, entropy: EntropyContext) -> list[dict]:
    examples=[row for row in allocation["examples"] if row.partition in {"clean_target_axis","clean_parallel_axis"}]
    by_category={category:[row for row in examples if _category(row)==category] for category in product(range(2),repeat=4)}
    pairs=[]; identities=set(); ordinal=0
    for stratum,count in STRATUM_COUNTS.items():
        each=count//16
        for category in sorted(by_category):
            bases=[]
            for row in by_category[category]:
                cf=_counterfactual(row,1-row.z2)
                if stratum in {"A","D"} and cf["answer"]==row.answer: continue
                if stratum=="B" and cf["answer"]!=row.answer: continue
                bases.append(row)
            bases.sort(key=lambda row:_score(entropy,"source-base-pair-allocation",[stratum,"base",row.example_id]))
            sources=list(by_category[_source_category(stratum,category)])
            sources.sort(key=lambda row:_score(entropy,"source-base-pair-allocation",[stratum,"source",row.example_id]))
            if len(bases)<each or len(sources)<each: raise StrictValidationError("PAIR_ELIGIBILITY",f"{stratum}:{category}")
            for offset in range(each):
                base,source=bases[offset],sources[offset]; cf=_counterfactual(base,source.z2)
                eligible={"A":base.z2!=source.z2 and cf["answer"]!=base.answer and base.answer!=source.answer,
                          "B":base.z2!=source.z2 and cf["answer"]==base.answer,
                          "C":base.z2==source.z2,
                          "D":base.z2!=source.z2 and base.answer==source.answer and cf["answer"]!=base.answer}[stratum]
                identity=(base.example_id,source.example_id)
                if not eligible or identity in identities or base.example_id==source.example_id: raise StrictValidationError("PAIR_ELIGIBILITY",str(identity))
                identities.add(identity); prefix="devonly-cppair-" if not entropy.claim_bearing else "exp33ae-cnfpair-"
                pair={"schema":"exp33ae.primary_pair.v1","namespace":entropy.namespace,"stage":entropy.stage,"claim_bearing":entropy.claim_bearing,
                    "pair_id":prefix+entropy.derive("source-base-pair-allocation",ordinal,16).hex(),"stratum":stratum,"eligible":True,
                    "base_id":base.example_id,"source_id":source.example_id,"base_programme_id":base.programme_id,"source_programme_id":source.programme_id,
                    "base_axis":base.axis,"source_axis":source.axis,"base_z2":base.z2,"base_z3":base.z3,"base_n3":base.n3,"base_answer":base.answer,
                    "source_z2":source.z2,"source_z3":source.z3,"source_n3":source.n3,"source_answer":source.answer,
                    "collision":cf["answer"]==base.answer,"same_observable_answer":base.answer==source.answer,"no_op":base.z2==source.z2,
                    "expected_z2":cf["z2"],"expected_z3":cf["z3"],"expected_n3":cf["n3"],"expected_answer":cf["answer"]}
                pair["pair_sha256"]=object_sha256(pair); pairs.append(pair); ordinal+=1
    if Counter(row["stratum"] for row in pairs)!=Counter(STRATUM_COUNTS) or len({(row["base_id"],row["source_id"]) for row in pairs})!=400:
        raise StrictValidationError("PAIR_ELIGIBILITY","exact strata/uniqueness")
    return pairs


def dev_only_fixture(root: Path) -> dict:
    from .entropy import dev_only_golden_context
    context=dev_only_golden_context(); allocation=allocate(root,context); pairs=build_pairs(allocation,context)
    return {"schema":"exp33ae.dev_only_allocation_fixture.v1","namespace":DEV_NAMESPACE,"stage":DEV_STAGE,"claim_bearing":False,
            "programmes":allocation["programmes"],"examples":[row.as_dict() for row in allocation["examples"]],"pairs":pairs,
            "balance":allocation["balance"],"development_exclusion_manifest":allocation["development_exclusion_manifest"],
            "confirmation_objects_created":0,"confirmation_objects_accessed":0}
