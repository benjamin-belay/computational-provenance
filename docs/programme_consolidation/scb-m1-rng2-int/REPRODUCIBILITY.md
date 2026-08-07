# Read-only reproducibility guide

This guide prioritizes auditing the retained prospective result. Reproduction
must not create replacement evidence, reopen protected qualification, or
expose protected confirmation identities.

Set:

```sh
REPO=/path/to/provenance-research
PYTHON=python3
cd "$REPO"
```

## Level 1 — audit retained evidence and digests

This is the recommended first-line reproduction.

```sh
"$PYTHON" \
  docs/programme_consolidation/scb-m1-rng2-int/verify_handoff.py \
  --repository .
```

Expected terminal:

```text
MAIN_CSW_AM_SCB_M1_RNG2_INT_HANDOFF_VERIFIED
```

The verifier is Python-standard-library-only. It:

- rejects duplicate JSON keys;
- checks every SHA-256 entry in the public artifact manifest;
- verifies the final programme terminal and closure boundary;
- verifies public INT-V1 counts and gates;
- verifies the protected 127/128 non-qualification;
- verifies calibrated `T` and `M`;
- verifies the 256-closure sealed confirmation attestation;
- checks the paper table source data against the retained audits.

## Level 2 — archival reconstruction boundary

Independent reconstruction from retained raw rows was performed in the source
research repository. Those raw rows, development tools, and model artifacts
are deliberately omitted from this curated GitHub edition because they are
large and include substantial superseded development history.

The compact claim-bearing audit outputs and their SHA-256 digests remain
available here. Level 1 is therefore self-contained; full row-level
reconstruction requires access to the separately retained archival research
workspace. This limitation does not turn a replay into new prospective
evidence.

## Level 3 — controlled regeneration boundary

Full regeneration includes model execution and stochastic generation. It is
not included in this curated edition and must occur only from the retained
archival source in a disposable clone or worktree. The frozen public source
commit is:

```text
87d9fda75fc175dffedfd2f29395118abf6a7450
```

The original exact claim-bearing source hashes were retained in:

```text
artifacts/dev_only/csw-am-scb-m1-rng2-int-v1/EXECUTABLE_MANIFEST.json
```

The original one-shot production command expects committed role material, key
files, and an empty destination. It must not be pointed at retained evidence.
An authorised auditor with archival access must:

1. create a disposable checkout at the frozen source commit;
2. copy the already-public release rows into a new stage;
3. restore the revealed public receipt and sampler keys at the frozen
   configured paths;
4. run `execute`, `reveal`, `evaluate`, and `finalize` once in order;
5. treat the regenerated result as a replay, not replacement evidence.

Do not use the controller release command: it requires the sealed allocation
secret and exists to enforce prospective release. Public retained release rows
were sufficient for the original replay but are not part of this compact
GitHub package.

## Figure and table reproduction

The paper-facing values are in:

```text
docs/programme_consolidation/scb-m1-rng2-int/results_summary.csv
docs/programme_consolidation/scb-m1-rng2-int/chronology.csv
```

They are deliberately plain CSV so figures can be regenerated without
experiment-specific Python.

The committed SVG figures can be regenerated with:

```sh
"$PYTHON" \
  docs/programme_consolidation/scb-m1-rng2-int/paper/generate_figures.py
```

## Known reproducibility boundaries

- The compact retained result is integrity-checkable; a new run is not
  prospective evidence.
- Row-level reconstruction and model regeneration require the separately
  retained archival workspace.
- Protected Q1 identities are already consumed and must not be rerun.
- Protected C1 identities are sealed and absent from this package.
- Full generation depends on the frozen model checkpoint and numerical
  environment.
- HMAC verification assumes trusted key custody.
- The handoff does not make an unconstrained-model or output-only claim.
