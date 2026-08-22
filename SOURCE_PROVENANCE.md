# Artifact provenance

The repository is a self-contained research release. `RELEASE_MANIFEST.json` records the size and SHA-256 digest of every released file; raw-evidence commitments bind the original uncompressed evidence streams.

| Release path | Role |
|---|---|
| `src/computational_provenance/` | Model definitions, scoring, receipt validation, and independent adjudication |
| `checkpoints/robustness/` | Five feed-forward and three engineered-transformer robustness checkpoints |
| `checkpoints/answer_only/` | Three answer-only checkpoints |
| `data/training/` | Exact registered training populations |
| `data/evaluation/robustness.jsonl.gz` | Registered 4,096-row held-out population |
| `evidence/components/` | Selected-model causal, receipt, calibration, and generic-channel evidence |
| `evidence/answer_only/` | Competence, split, fitted probes, site evaluation, and terminal evidence |
| `evidence/full/` | Raw and adjudicated public and protected end-to-end evidence |
| `protocols/` | Portable frozen protocols and scientific gates |

Deterministic gzip files are hashed over their uncompressed bytes when they correspond to prospective raw-evidence commitments.
