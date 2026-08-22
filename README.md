# Towards Computational Provenance

This is the research release for **Towards Computational Provenance: Carrying Causal-State Evidence in Generated Text**. It contains the code, frozen protocols, checkpoints, held-out population, probe coefficients, and original audit evidence needed to reproduce every result reported in the paper.

## Paper results represented here

| Claim-bearing result | Released result |
|---|---:|
| Hard-bottleneck feed-forward computation and causal robustness | 5/5 models |
| Engineered-transformer computation and causal robustness | 3/3 models |
| Feed-forward public / protected end-to-end evaluation | 128/128 / 128/128 |
| Transformer public / protected end-to-end evaluation | 128/128 / 128/128 |
| Answer-only transformer competence / registered-site localisation | 3/3 / 0/20 |

Each robustness model is evaluated on the same 4,096 held-out rows and 28,672 causal cases. Each end-to-end evaluation contains 10,240 generated outputs: eight outputs for each of five conditions (`STATE`, `CARRIER_OFF`, `PROMPT`, `ANSWER`, and `GENERIC`) across 256 natural/intervened execution packages. The retained outputs use the frozen 49-step generator trace with 24 carrier-eligible positions.

Superseded runs—including the older 127/128 result and lower development experiments—are intentionally absent. No claim in the paper depends on them.

## Quick start

Python 3.11 or newer is required; Python 3.12.13 is the frozen experiment version.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python scripts/verify_release.py
```

The principal reproduction commands are:

```bash
python scripts/reproduce_robustness.py all
python scripts/reproduce_answer_only.py
python scripts/verify_evidence.py
```

The first two recompute the model-level paper results from released checkpoints. The third independently verifies the original four end-to-end evaluations: uncompressed raw-evidence commitments, all revealed-key HMAC receipts, output and control denominators, adjudication artifacts, independent reference results, and terminal hash chains.

See [REPRODUCE.md](REPRODUCE.md) for the complete workflow, [PAPER_RESULTS.md](PAPER_RESULTS.md) for claim-to-artifact mapping, and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for research-source provenance.

## Scope

This is a controlled proof of concept, not output-only proof of an exact execution. Exact execution claims require an authenticated receipt; the text carrier identifies an authenticated abstract state class. The result uses engineered hard bottlenecks, a trusted receipt issuer, and a constrained generator. It does not establish naturally emerging provenance in ordinary pretrained language models.

## License

MIT. See [LICENSE](LICENSE).
