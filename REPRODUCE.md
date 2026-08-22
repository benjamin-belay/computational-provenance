# Reproduction guide

## Environment

The frozen runs used Python 3.12.13, PyTorch 2.7.1, and NumPy 2.2.6. Those package versions are pinned in `pyproject.toml`; CPU evaluation is supported.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

## One-command release check

```bash
python scripts/verify_release.py
```

This validates every file in `RELEASE_MANIFEST.json`, then checks the original raw evidence and four final terminal chains.

## Recompute the paper's model results

```bash
python scripts/reproduce_robustness.py all
```

This loads five frozen feed-forward checkpoints and three frozen engineered-transformer checkpoints. For each model it evaluates all 4,096 held-out rows under `NATURAL`, `ALIAS_Z2`, `NON_ALIAS_Z2`, `WRONG_Z2`, `SAME_Z2`, `SHAM`, and `DIRECT_Z3`: 28,672 cases per model. It applies the prospectively frozen gates and reproduces 5/5 and 3/3.

```bash
python scripts/reproduce_answer_only.py
```

This evaluates all three answer-only checkpoints on the 4,096-row validation split, then loads only the registered development model, recreates the disjoint 3,072-row site-selection activations, verifies their exact byte digests, and evaluates the 260 frozen probe fits across 20 sites. It reproduces 3/3 competence, 0/20 qualifying sites, best full-state accuracy 23.80%, and best quotient accuracy 45.54%.

## Verify the original end-to-end evaluations

```bash
python scripts/verify_evidence.py
```

For feed-forward and transformer, public and protected, this verifies:

- the SHA-256 commitments to the five uncompressed raw evidence streams;
- all 1,280 HMAC receipt envelopes against the key revealed only after the raw commitment;
- 256 execution packages, 1,280 generation authorizations, and 10,240 outputs;
- exactly 2,048 outputs for each of the five conditions;
- 49 model steps and 24 eligible positions per output;
- all production-artifact hashes, independent reference results, and terminal audit hashes;
- 128/128 complete closures in each of the four evaluations.

This is the exact prospective evidence path. The smaller `scripts/replay.py` remains a readable diagnostic implementation that issues fresh deterministic keys; it is not used to establish the original chronology or output denominator.

## Diagnostic replay and retraining

```bash
python scripts/replay.py --architecture feedforward --split public --limit 4
python scripts/replay.py --architecture transformer --split protected --limit 4
```

Exact fixed training populations and recipes are also included:

```bash
python scripts/train.py --architecture feedforward --output replay_output/feedforward
python scripts/train.py --architecture transformer --output replay_output/transformer
python scripts/replay.py --architecture feedforward --split public --checkpoint replay_output/feedforward/checkpoint.pt
```

The transformer recipe has 12,000 CPU steps and may take hours. Repeated training can differ across hardware kernels even with deterministic settings; the paper's claim is therefore bound to the included checkpoints and hashes, while retraining is a recipe replication check.
