# Paper claim and artifact map

Only the final claim-bearing lineage is released.

| Paper result | Reproduction entry point | Frozen artifacts |
|---|---|---|
| Feed-forward 5/5 robustness | `scripts/reproduce_robustness.py feedforward` | `protocols/feedforward_robustness.json`, `checkpoints/robustness/feedforward/`, `data/evaluation/robustness.jsonl.gz` |
| Engineered-transformer 3/3 robustness | `scripts/reproduce_robustness.py transformer` | `protocols/transformer_robustness.json`, `checkpoints/robustness/transformer/`, same held-out population |
| Answer-only 3/3 competence | `scripts/reproduce_answer_only.py` | `protocols/answer_only_competence.json`, `checkpoints/answer_only/`, `evidence/answer_only/competence/` |
| Answer-only 0/20 localisation | `scripts/reproduce_answer_only.py` | `protocols/answer_only_localisation.json`, development checkpoint 1, frozen coefficient archive and site results |
| Feed-forward 128/128 public and protected | `scripts/verify_evidence.py` | `evidence/full/feedforward/{public,protected}/` and bound protocols |
| Transformer 128/128 public and protected | `scripts/verify_evidence.py` | `evidence/full/transformer/{public,protected}/` and bound protocols |

The full end-to-end directories retain raw outputs and attack inputs as deterministic gzip files. Hashes are always computed over their original uncompressed bytes so they match the prospective commitments.

The answer-only result is a scoped negative: none of 20 registered token sites passed the frozen linear-recoverability gates. It does not claim that the transformer contains no representation of the intermediate state by any possible analysis.
