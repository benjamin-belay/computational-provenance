# Publication figures

These SVGs are presentation-layer renderings of the frozen reporting record.
They add no scientific evidence and do not modify any threshold, terminal,
denominator, protected-data status, or historical artefact.

Regenerate them from the repository root with:

```bash
.venv/bin/python \
  docs/programme_consolidation/scb-m1-rng2-int/paper/generate_figures.py
```

`FIGURE_MANIFEST.json` binds the source and output SHA-256 digests. Sources are
the committed `results_summary.csv`, `chronology.csv`, and paper draft. The
generator uses only the Python standard library and emits accessible,
publication-sized SVG.

## Files

- `figure-1-causal-computation.svg`: mandatory learned bottleneck.
- `figure-2-receipt-gated-carriage.svg`: verification-before-authority path.
- `figure-3-matched-contrast.svg`: fixed and switching variables.
- `figure-4-evidence-hierarchy.svg`: chronology-derived evidence classes.
- `table-3-public-protected-outcomes.svg`: separate public, protected
  qualification, and protected confirmation outcomes.

All figures concern selected Model 1 with a supervised mandatory bottleneck,
trusted HMAC receipts, and an engineered RNG2 realiser. Public validation and
protected non-qualification remain distinct.
