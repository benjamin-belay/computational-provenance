# Reproduction guide

There are three different activities that should not be conflated.

## 1. Verify the retained Paper 1 result

This is self-contained and uses only the Python standard library:

```bash
python docs/programme_consolidation/scb-m1-rng2-int/verify_handoff.py --repository .
```

It validates the public artifact manifest, reported public and protected
counts, frozen thresholds, closure status, and table source data.

## 2. Exercise the released implementation

Create the locked Python 3.12 environment and run the focused tests:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest tests/test_csw_am_scb_m1.py
pytest tests/test_csw_am_scb_m1_rec_d0.py
pytest tests/test_csw_am_scb_m1_rng2_d0.py
pytest tests/test_csw_am_scb_m1_rng2_int_v1.py
```

The selected checkpoint and the disclosed public evaluation population are
included at their original repository-relative paths.

## 3. Generate a new replay

The one-shot production scripts preserve the frozen research configuration,
including original controller paths and lifecycle restrictions. They must not
be pointed at the retained evidence directories. A new replay requires fresh
keys, a disposable output directory, and explicit adaptation of controller
paths to the new environment.

A replay is useful for implementation study but is not replacement prospective
evidence. Do not rerun protected qualification, release sealed confirmation
identities, or present a replay as an extension of the frozen result.

## Paper figures

Regenerate the archival paper-facing SVGs with:

```bash
python docs/programme_consolidation/scb-m1-rng2-int/paper/generate_figures.py
```

The ICLR LaTeX source is under `paper/`.
