"""Bounded one-access controller with locking, durable hash chain, exact resume."""

from __future__ import annotations

from hashlib import sha256
import fcntl,json,os
from pathlib import Path

from ..remediation.strict import canonical_json,object_sha256,strict_load_json,strict_load_jsonl,StrictValidationError


class AccessViolation(StrictValidationError):
    def __init__(self,detail): super().__init__("ACCESS_STATE",detail)
class ConcurrentAccess(AccessViolation): pass


def fsync_directory(path:Path)->None:
    descriptor=os.open(path,os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def atomic_json(path:Path,value:dict)->None:
    temporary=path.with_suffix(path.suffix+".pending");data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode();descriptor=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try: os.write(descriptor,data);os.fsync(descriptor)
    finally: os.close(descriptor)
    os.replace(temporary,path);fsync_directory(path.parent)


class AccessController:
    def __init__(self,directory:Path,namespace:str,stage:str,claim_bearing:bool,token:bytes,registry_commitment:str,expected_call_ids:list[str]):
        if type(token) is not bytes or len(token)!=32 or type(claim_bearing) is not bool or len(expected_call_ids)!=19600 or len(set(expected_call_ids))!=19600: raise AccessViolation("invalid controller inputs")
        self.directory=directory;self.namespace=namespace;self.stage=stage;self.claim_bearing=claim_bearing;self.token_hash=sha256(token).hexdigest();self.registry_commitment=registry_commitment;self.expected=tuple(expected_call_ids)
        self.lock_path=directory/"access.lock";self.state_path=directory/"state.json";self.journal_path=directory/"journal.jsonl";self.buffer_path=directory/"buffer.jsonl";self.lock=None

    def acquire(self,initialize=False):
        self.directory.mkdir(parents=True,exist_ok=True);descriptor=os.open(self.lock_path,os.O_RDWR|os.O_CREAT,0o600)
        try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error: os.close(descriptor);raise ConcurrentAccess("concurrent access rejected") from error
        self.lock=descriptor
        if initialize:
            if any(path.exists() for path in (self.state_path,self.journal_path,self.buffer_path)): self.close();raise AccessViolation("access state already exists")
            state={"schema":"exp33ae.access_state.v1","namespace":self.namespace,"stage":self.stage,"claim_bearing":self.claim_bearing,"state":"UNOPENED","token_sha256":self.token_hash,"registry_commitment":self.registry_commitment,"expected_calls":19600,"buffered_calls":0,"journal_events":0,"journal_tail_sha256":"0"*64}
            descriptor_state=os.open(self.state_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            try: os.write(descriptor_state,(json.dumps(state,indent=2,sort_keys=True)+"\n").encode());os.fsync(descriptor_state)
            finally: os.close(descriptor_state)
            fsync_directory(self.directory)
        self.validate();return self

    def close(self):
        if self.lock is not None: fcntl.flock(self.lock,fcntl.LOCK_UN);os.close(self.lock);self.lock=None
    def __enter__(self):
        if self.lock is None: raise AccessViolation("not acquired")
        return self
    def __exit__(self,*_): self.close()

    def _state(self):
        value=strict_load_json(self.state_path);keys={"schema","namespace","stage","claim_bearing","state","token_sha256","registry_commitment","expected_calls","buffered_calls","journal_events","journal_tail_sha256"}
        if frozenset(value) not in {frozenset(keys),frozenset(keys|{"invalid_reason"})} or value.get("schema")!="exp33ae.access_state.v1" or value.get("namespace")!=self.namespace or value.get("stage")!=self.stage or value.get("claim_bearing") is not self.claim_bearing or value.get("token_sha256")!=self.token_hash or value.get("registry_commitment")!=self.registry_commitment or value.get("expected_calls")!=19600 or value.get("state") not in {"UNOPENED","BUFFERING","BUFFER_COMPLETE","RELEASED","INVALID"}: raise AccessViolation("state schema/binding")
        return value
    def _write_state(self,value): atomic_json(self.state_path,value)
    def _invalidate(self,reason):
        try: state=strict_load_json(self.state_path);state["state"]="INVALID";state["invalid_reason"]=reason;atomic_json(self.state_path,state)
        except Exception: pass

    def validate(self):
        state=self._state();events=strict_load_jsonl(self.journal_path) if self.journal_path.exists() else [];previous="0"*64
        details={"ONE_ACCESS_CONSUMED":{"registry_commitment","expected_calls"},"RESUME_FROM_FIRST_MISSING":{"first_missing_order"},"CALLS_BUFFERED":{"start","count","end","buffer_sha256"},"BUFFER_COMPLETE":{"calls","buffer_sha256"},"EVALUATOR_DERIVED_ATOMIC_RELEASE":{"release_sha256","status"},"SECOND_ACCESS_REJECTED":{"requested_resume"}}
        for ordinal,row in enumerate(events):
            copy=dict(row);digest=copy.pop("event_sha256",None)
            if set(copy)!={"schema","namespace","stage","claim_bearing","ordinal","event","details","previous_event_sha256"} or copy["schema"]!="exp33ae.access_event.v1" or copy["namespace"]!=self.namespace or copy["stage"]!=self.stage or copy["claim_bearing"] is not self.claim_bearing or copy["ordinal"]!=ordinal or copy["previous_event_sha256"]!=previous or digest!=object_sha256(copy) or copy["event"] not in details or set(copy["details"])!=details[copy["event"]]: self._invalidate("ACCESS_CHAIN");raise AccessViolation("journal chain")
            previous=digest
        if state["journal_events"]!=len(events) or state["journal_tail_sha256"]!=previous: self._invalidate("ACCESS_CHAIN");raise AccessViolation("journal truncation/reordering")
        rows=strict_load_jsonl(self.buffer_path) if self.buffer_path.exists() else [];ids=[row.get("call_id") for row in rows]
        if tuple(ids)!=self.expected[:len(ids)] or len(ids)!=len(set(ids)) or state["buffered_calls"]!=len(rows): self._invalidate("CALL_RECONCILIATION");raise AccessViolation("buffer exact-prefix failure")
        return {"state":state,"events":events,"buffer_rows":rows}

    def _event(self,event,details):
        state=self._state();row={"schema":"exp33ae.access_event.v1","namespace":self.namespace,"stage":self.stage,"claim_bearing":self.claim_bearing,"ordinal":state["journal_events"],"event":event,"details":details,"previous_event_sha256":state["journal_tail_sha256"]};row["event_sha256"]=object_sha256(row)
        descriptor=os.open(self.journal_path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try: os.write(descriptor,(canonical_json(row)+"\n").encode());os.fsync(descriptor)
        finally: os.close(descriptor)
        fsync_directory(self.directory);state["journal_events"]+=1;state["journal_tail_sha256"]=row["event_sha256"];self._write_state(state)

    def begin(self,resume=False):
        state=self.validate()["state"]
        if state["state"]=="UNOPENED" and not resume:
            self._event("ONE_ACCESS_CONSUMED",{"registry_commitment":self.registry_commitment,"expected_calls":19600});state=self._state();state["state"]="BUFFERING";self._write_state(state);return 0
        if state["state"]=="BUFFERING" and resume:
            self._event("RESUME_FROM_FIRST_MISSING",{"first_missing_order":state["buffered_calls"]});return state["buffered_calls"]
        self._event("SECOND_ACCESS_REJECTED",{"requested_resume":resume});self._invalidate("SECOND_ACCESS");raise AccessViolation("second/out-of-order access")

    def append(self,rows):
        state=self.validate()["state"];start=state["buffered_calls"];ids=[row.get("call_id") for row in rows]
        if state["state"]!="BUFFERING" or not rows or tuple(ids)!=self.expected[start:start+len(rows)] or len(ids)!=len(set(ids)): self._invalidate("CALL_RECONCILIATION");raise AccessViolation("not exact next segment")
        descriptor=os.open(self.buffer_path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
        try:
            for row in rows: os.write(descriptor,(canonical_json(row)+"\n").encode())
            os.fsync(descriptor)
        finally: os.close(descriptor)
        fsync_directory(self.directory);state["buffered_calls"]+=len(rows);self._write_state(state);self._event("CALLS_BUFFERED",{"start":start,"count":len(rows),"end":start+len(rows),"buffer_sha256":sha256(self.buffer_path.read_bytes()).hexdigest()})

    def complete(self):
        state=self.validate()["state"]
        if state["state"]!="BUFFERING" or state["buffered_calls"]!=19600: raise AccessViolation("incomplete buffer")
        self._event("BUFFER_COMPLETE",{"calls":19600,"buffer_sha256":sha256(self.buffer_path.read_bytes()).hexdigest()});state=self._state();state["state"]="BUFFER_COMPLETE";self._write_state(state)

    def record_release(self,release_sha256,status):
        if self.validate()["state"]["state"]!="BUFFER_COMPLETE": raise AccessViolation("release before complete")
        self._event("EVALUATOR_DERIVED_ATOMIC_RELEASE",{"release_sha256":release_sha256,"status":status});state=self._state();state["state"]="RELEASED";self._write_state(state)
