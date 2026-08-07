# Can Generated Text Carry Evidence of an Authenticated Causal State?

This repository contains the code, frozen protocols, public artifacts, paper
source, and audit material for Paper 1 of the computational-provenance research
programme.

The paper studies a bounded feasibility question: can an authenticated,
causally operative internal state select a statistical signal in generated
text while the prompt, final answer, semantic payload, generation settings,
and primitive random draws remain paired?

## Result boundary

- Public receipt-bound integration: **128/128** complete closures passed.
- Protected qualification: **127/128**; qualification was not earned.
- Final protected confirmation: not run.

This is evidence for one selected engineered model with a supervised mandatory
hard bottleneck and trusted receipt infrastructure. It is not output-only
authentication, whole-reasoning provenance, or evidence of protected
generalisation.

## What is included

- `src/exp33ae/`: reusable model, task, intervention, evaluation, and audit code.
- `tools/`: the frozen selected-model, receipt, RNG2 carrier, calibration, and
  integration implementations and independent references.
- `protocol/`: the corresponding frozen JSON protocols and configurations.
- `tests/`: focused tests for the Paper 1 implementation family.
- `artifacts/`: the selected 12,648-parameter checkpoint, public evaluation
  population, and compact claim-bearing audit records.
- `docs/programme_consolidation/scb-m1-rng2-int/`: result, limitations,
  chronology, source data, review materials, and integrity verifier.
- `paper/`: the anonymous ICLR 2027 LaTeX source and figures.

Protected identities, unrevealed keys, controller workspaces, virtual
environments, caches, and superseded development traces are intentionally not
included.

## Quick verification

Use Python 3.12. The recorded environment is pinned in
`requirements-lock.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python docs/programme_consolidation/scb-m1-rng2-int/verify_handoff.py --repository .
pytest tests/test_csw_am_scb_m1_rng2_int_v1.py
```

The integrity command should end with:

```text
MAIN_CSW_AM_SCB_M1_RNG2_INT_HANDOFF_VERIFIED
```

See [REPRODUCE.md](REPRODUCE.md) for the distinction between auditing the
retained result, exercising the released implementation, and generating a new
replay.

## Frozen source

The public integration was frozen at source commit
`87d9fda75fc175dffedfd2f29395118abf6a7450`. Files copied from that commit have
not been rewritten. Publication-only files such as this README, citation
metadata, and the final paper source were added later and do not alter the
recorded scientific evidence.

## Citation and license

Citation metadata is provided in `CITATION.cff`. Code is released under the
MIT License. Paper text and third-party conference style files may have their
own applicable terms.
