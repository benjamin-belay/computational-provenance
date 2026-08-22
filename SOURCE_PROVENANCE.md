# Source provenance

This release was curated from the final claim-bearing artifacts present in the research repository at commit:

```text
f414be685c507bc88493622829734d358dab27f4
```

Individual prospective protocols retain their own lineage commits and source hashes. `RELEASE_MANIFEST.json` separately binds every released byte.

| Release path | Frozen research source |
|---|---|
| `src/computational_provenance/feedforward.py` | final strict-bottleneck architecture from `tools/csw_am_scb_m1.py` |
| `src/computational_provenance/transformer.py` | `src/exp33atreng/model.py` |
| `src/computational_provenance/robustness.py` | registered FF R2-B0 and TR-ENG-R1 evaluation semantics |
| `src/computational_provenance/answer_only.py` | `tools/csw_am_ls_f0.py` and `tools/csw_am_tr_v2_l1.py` |
| `checkpoints/robustness/` | five R2 step-5000 and three TR-ENG-R1 step-12000 claim checkpoints |
| `checkpoints/answer_only/` | three answer-only step-12000 claim checkpoints |
| `data/evaluation/robustness.jsonl.gz` | registered 4,096-row held-out population, original SHA-256 `f2ffd11c...` |
| `evidence/answer_only/` | competence, split, frozen probes, 20-site evaluation, and terminal evidence |
| `evidence/full/` | raw and adjudicated P2/Q1 evidence for both architectures |
| `protocols/` | exact final robustness, answer-only, public, protected, and receipt protocols |

The older summary directories under `evidence/feedforward/` and `evidence/transformer/` are retained for compatibility with the initial clean release. The complete claim evidence is under `evidence/full/`; neither location contains the superseded 127/128 experiment.

`evidence/transformer/protected/ADJUDICATION_REPAIR.json` is an incident record from the final positive evaluation. It documents a post-evidence schema alias and confirms that no model run, output, receipt, key, threshold, or gate changed.
