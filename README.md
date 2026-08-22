# Computational provenance

This repository is the clean research release for **Receipt-Gated Causal-State Carriage**. It demonstrates a bounded form of computational provenance: a verified causal state inside a model selects a subtle, detectable pattern in generated text.

The final result is reproduced in two deliberately controlled architectures:

| Architecture | Parameters | Public evaluation | Fresh protected evaluation |
|---|---:|---:|---:|
| Hard-bottleneck feed-forward network | 12,648 | 128/128 paired cases | 128/128 paired cases |
| Two-stage hard-bottleneck transformer | 546,088 | 128/128 paired cases | 128/128 paired cases |

For each case, the natural and intervened executions have the same prompt, answer, semantic payload, and position-indexed random draws. They differ in the causal `z2 -> z3` path. An authenticated `z2` category selects one of 16 carrier authorities; the detector later recovers the corresponding authority from the generated report.

## What is included

- the exact final feed-forward and transformer checkpoints;
- the 128 public and 128 released protected evaluation rows for each architecture;
- a readable implementation of the task, both architectures, HMAC receipts, paired sampler, text carrier, detector, and end-to-end replay;
- final public/protected audit summaries and terminals;
- tests for the no-bypass structure, receipt rejection, paired randomness, carrier detection, and evidence hashes.

Superseded experiments, including the earlier 127/128 development lineage, are intentionally not included. The retained evidence reports only the final prospectively trained models and their 128/128 evaluations.

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
pytest
python scripts/verify_release.py
python scripts/replay.py --architecture feedforward --split public
python scripts/replay.py --architecture transformer --split public
```

The complete 128-case replay generates 4,096 reports per architecture (natural and alternative executions, eight reports per state, plus carrier-off controls), so it takes longer than the unit tests. Use `--limit 4` for a quick smoke test.

## Boundaries

This is a controlled proof of concept, not output-only proof of an exact execution. Exact execution claims require the authenticated receipt; the text carrier identifies the authenticated abstract state class. The models use engineered hard bottlenecks, the receipt issuer is trusted, and the text generator is constrained. The result does not establish naturally emerging provenance in ordinary transformers or pretrained language models.

See [REPRODUCE.md](REPRODUCE.md) for commands and [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md) for the mapping to the frozen research evidence.

## License

MIT. See [LICENSE](LICENSE).
