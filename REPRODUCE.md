# Reproduction guide

## 1. Verify the retained release

```bash
python scripts/verify_release.py
```

This checks the SHA-256 of the final checkpoints, evaluation populations, audits, and terminals. It also verifies that all four retained terminals report 128/128 complete paired cases.

## 2. Run the tests

```bash
pytest
```

The tests cover the arithmetic relation, model sizes, checkpoint identity, hard-state intervention, HMAC rejection, paired randomness, detector recovery, and a four-case end-to-end replay for each architecture.

## 3. Replay an evaluation

```bash
python scripts/replay.py --architecture feedforward --split public
python scripts/replay.py --architecture feedforward --split protected
python scripts/replay.py --architecture transformer --split public
python scripts/replay.py --architecture transformer --split protected
```

For a quick check:

```bash
python scripts/replay.py --architecture transformer --split public --limit 4
```

The replay verifies the checkpoint's natural and forced `z2+8` computations, issues and verifies fresh deterministic HMAC receipts, derives the state-conditioned carrier, generates eight reports per side, and scores all 16 possible authorities. It also runs a carrier-off control.

## Prospective evidence versus replay

The JSON files under `evidence/` are the retained summaries from the original one-shot evaluations. The replay uses fresh deterministic research keys so that it is safe and repeatable; it is a reproduction check, not a replacement for the original prospective chronology. Original secret key material is not published.

## Hardware

Checkpoint evaluation is CPU-compatible. Full retraining is deliberately not the default reproduction path: it is more expensive, while the claim-bearing checkpoints and evaluation rows are already frozen and included.

The exact training populations and fixed recipes are nevertheless included. To retrain:

```bash
python scripts/train.py --architecture feedforward --output replay_output/feedforward-training
python scripts/train.py --architecture transformer --output replay_output/transformer-training
```

The transformer recipe uses 12,000 CPU training steps and can take substantial time. Retraining produces a new reproduction artifact; the published claim remains bound to the included checkpoint hashes.
