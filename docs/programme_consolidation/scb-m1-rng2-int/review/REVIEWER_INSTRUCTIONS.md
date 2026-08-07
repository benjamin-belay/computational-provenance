# Independent reviewer instructions

## Review object

Review repository commit:

```text
15e25785eee560cf0242ff781dedb4ea4fc0e809
```

The exact files and digests are fixed in
[REVIEW_MANIFEST.json](REVIEW_MANIFEST.json). Review that commit, not the
current branch tip. Later files are reporting drafts and are not evidence.

The published SHA-256 of the review manifest is recorded in
[REVIEW_RELEASE.json](REVIEW_RELEASE.json). Verify it before beginning.

The distinction between the immutable evidence object and mutable
reporting-only layer is defined in
[EVIDENCE_FREEZE_AND_REPORTING_POLICY.md](EVIDENCE_FREEZE_AND_REPORTING_POLICY.md).

## Independence disclosure

State whether you:

- wrote or reviewed implementation code;
- helped select thresholds, gates, attacks, or terminal rules;
- saw experimental outcomes before they were frozen;
- have another relationship that could affect independence.

Prior involvement does not automatically disqualify the review, but it must be
visible.

## Requested output

Use the frozen
[independent-review template](../INDEPENDENT_REVIEW_TEMPLATE.md). Record
findings rather than issuing a generic certification:

- `CRITICAL`: invalidates or materially narrows the central claim;
- `MAJOR`: requires a claim or methodological correction;
- `MINOR`: reporting, clarity, or reproducibility problem;
- `ACCEPTED`: evidence supports the stated bounded claim.

For every finding, cite the exact commit, path, retained artifact, and—where
applicable—command output.

## Central questions

Assess separately:

> Does the archive support a bounded public demonstration that a statistical
> output signal tracks an authenticated, causally operative abstract state
> while prompt, answer, semantic payload, and primitive randomness remain
> fixed?

and:

> Does the archive establish protected generalisation or final protected
> confirmation?

The second question must not inherit the public result.

## Protected Q1 wording

Judge whether the archive supports this precise formulation:

> Q1's release, execution chronology, denominators, retained evidence, and
> independent reconstruction were valid. Scientifically, it failed one
> computation-fidelity gate and two dependent attack expectations; the frozen
> adjudicator then issued an invalid terminal because it lacked an appropriate
> scientific-negative outcome.

Do not reinterpret Q1 as either a protected pass or meaningless evidence.

## Commands

Run only the read-only and retained-evidence commands listed in the review
manifest. A full public replay is optional and requires a separate declaration
because it is a reproducibility exercise, not prospective replacement
evidence.

Do not:

- access or request sealed confirmation identities;
- rerun protected Q1;
- modify historical artifacts;
- repair or relabel a historical terminal;
- select new claims from exploratory analyses.

## Report provenance

Include:

- reviewer identity or stable anonymised code;
- review date;
- reviewed commit;
- SHA-256 of `REVIEW_MANIFEST.json`;
- commands and environment;
- all findings and unresolved questions;
- final assessment of each claim;
- optional signature, or at minimum the SHA-256 of the completed report.
