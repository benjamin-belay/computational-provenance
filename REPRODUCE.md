# Reproduction guide

## Environment

The frozen runs used Python 3.12.13, PyTorch 2.7.1, NumPy 2.2.6, and scikit-learn 1.7.1. Those package versions are pinned in `pyproject.toml`; CPU evaluation is supported.

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

This evaluates all three answer-only checkpoints on the 4,096-row validation split, then loads only the registered development model, recreates the disjoint 3,072-row site-selection activations, and evaluates the 260 frozen probe fits across 20 sites. It reproduces 3/3 competence, 0/20 qualifying sites, best full-state accuracy 23.80%, and best quotient accuracy 45.54%.

The output also reports whether activation arrays match the frozen byte digests. Exact floating-point bytes can differ across operating systems and numerical kernels even when every discrete prediction and paper result agrees. On a matching numerical platform, enforce byte identity with `--require-exact-activations`; cross-platform reproduction uses the registered decisions and metrics.

## Recompute the selected-model component experiments

```bash
python scripts/reproduce_components.py
```

This reruns the 512-example causal confirmation, validates 512 genuine receipt packages and adjudicates all 9,216 receipt attacks, derives the carrier threshold and margin from 16,384 calibration outputs, and reconstructs the 512-closure generic carrier validation from another 16,384 outputs.

## Recompute the end-to-end evaluations

```bash
python scripts/reproduce_end_to_end.py
```

This reconstructs all four public/protected evaluations from raw evidence. To run one evaluation, pass both `--architecture` and `--split`.

## Verify evidence integrity

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

Exact fixed training populations and recipes are also included:

```bash
python scripts/train.py --architecture feedforward --profile robustness --output reproduced/feedforward-robustness
python scripts/train.py --architecture feedforward --profile designated --output reproduced/feedforward-designated
python scripts/train.py --architecture transformer --profile robustness --output reproduced/transformer-robustness
python scripts/train.py --architecture transformer --profile designated --output reproduced/transformer-designated
python scripts/train_answer_only.py --output reproduced/answer-only
python scripts/train_answer_only.py --fit-probes --output reproduced/probe-coefficients.npz
```

The transformer recipes have 12,000 CPU steps and may take hours. Use `--steps 2` for a wiring smoke test; that override does not reproduce a paper checkpoint. Repeated training can differ across hardware kernels even with deterministic settings, so checkpoint hashes are recorded alongside metric-level reproduction.
