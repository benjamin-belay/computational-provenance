# Figure and table plan

All numerical values must be generated from `../results_summary.csv` and
`../chronology.csv`, or cited directly from the frozen reviewed commit.
Publication SVGs are generated deterministically with:

```text
.venv/bin/python \
  docs/programme_consolidation/scb-m1-rng2-int/paper/generate_figures.py
```

The generator writes a source-and-output digest manifest to
`figures/FIGURE_MANIFEST.json`. It is a presentation-layer tool and must not
write to the frozen evidence object.

## Figure 1 — mandatory causal computation

Generated asset:
[figure-1-causal-computation.svg](figures/figure-1-causal-computation.svg).

Purpose: show the mandatory `z2 → z3 → answer` path without overloading the
figure with receipt and carrier governance.

## Figure 2 — receipt-gated state carriage

Generated asset:
[figure-2-receipt-gated-carriage.svg](figures/figure-2-receipt-gated-carriage.svg).

Purpose: show exact-event and abstract-state evidence, chain verification,
verification-before-authority ordering, RNG2 generation, and 16-authority
scoring.

## Figure 3 — matched natural/alias contrast

Use a two-column schematic:

```text
fixed: prompt, answer, payload, configuration, indexed draws
switches: authenticated z2, state authority, statistical carrier
```

Do not imply that exact execution identity is carried statistically.

Generated asset:
[figure-3-matched-contrast.svg](figures/figure-3-matched-contrast.svg).

## Table 1 — exact versus abstract provenance

Source: the decision table in `PAPER_DRAFT.md`.

Purpose: prevent conflation of exact receipts with abstract carrier
compatibility.

## Table 2 — experimental chronology

Source: `../chronology.csv`.

Include valid negatives and invalid stages, not only the successful chain.

## Table 3 — public and protected outcomes

Source: `../results_summary.csv`.

Display public 128/128 and protected 127/128 on separate rows with explicit
formal statuses. Never pool them.
Generated asset:
[table-3-public-protected-outcomes.svg](figures/table-3-public-protected-outcomes.svg).

## Figure 4 — evidence hierarchy

Suggested categories:

```text
formal positives
valid scientific negatives
implementation/chronology invalidity
descriptive diagnostics
sealed/unexecuted evidence
```

This is a reporting figure, not a quantitative meta-analysis.

Generated asset:
[figure-4-evidence-hierarchy.svg](figures/figure-4-evidence-hierarchy.svg).

The full governance chronology belongs in `APPENDICES.md`; the main paper
should retain only the public/protected result table and the minimal successful
chain.

## Caption requirements

Every relevant caption must state:

- selected Model 1;
- supervised mandatory bottleneck;
- trusted HMAC receipts;
- engineered RNG2 realiser;
- public validation versus protected non-qualification.

## Reproduction

The five SVGs are deterministically produced by
[`generate_figures.py`](generate_figures.py). Their source and output digests
are recorded in
[`FIGURE_MANIFEST.json`](figures/FIGURE_MANIFEST.json). The generator is
presentation-only and does not access sealed or protected identities.
