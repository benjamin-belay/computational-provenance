# Source provenance

This release was curated from research repository commit:

```text
b30c640465c4ce895a58ced2752561906c1325a6
```

The cleanup removes superseded protocol layers and historical 127/128 development results. It does not alter the final checkpoints, evaluation rows, audit summaries, or terminal records.

| Release path | Frozen research source |
|---|---|
| `src/computational_provenance/feedforward.py` | architecture extracted from `tools/csw_am_scb_m1.py` |
| `src/computational_provenance/transformer.py` | byte-for-byte copy of `src/exp33atreng/model.py` |
| `src/computational_provenance/carrier.py` | readable extraction of `tools/csw_a0_stochastic_channel.py`, `tools/csw_am_scb_m1_rng2_d0.py`, and `tools/csw_am_scb_m1_rng2_cal1.py` |
| `src/computational_provenance/receipts.py` | readable release implementation of the HMAC receipt and chain semantics |
| `checkpoints/feedforward.pt` | `artifacts/dev_only/csw-am-scb-r3-t1/training/claim_checkpoint/R3_MODEL_1_STEP_5000.pt` |
| `checkpoints/transformer.pt` | `artifacts/dev_only/csw-am-tr-eng-t1/training/checkpoint/TR_ENG_PROVENANCE_MODEL_1_STEP_12000.pt` |
| `data/feedforward/*` | final R3 public/protected registered rows |
| `data/transformer/*` | final TR-ENG public/protected registered rows |
| `evidence/feedforward/*` | final R3 P2/Q1 audits and terminals |
| `evidence/transformer/*` | final TR-ENG P2/Q1 audits and terminals |

The clean replay code is intentionally easier to read than the lifecycle controller used for the one-shot runs. `RELEASE_MANIFEST.json` binds the immutable claim-bearing files. The original evaluation incident record is retained at `evidence/transformer/protected/ADJUDICATION_REPAIR.json`; it describes a post-evidence schema alias and confirms that no model run, output, receipt, key, threshold, or gate changed.
