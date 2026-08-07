# Evidence freeze and reporting policy

The evidence object and reporting layer have different change rules.

## Immutable evidence object

The independent review object is repository commit:

```text
15e25785eee560cf0242ff781dedb4ea4fc0e809
```

Its review-manifest SHA-256 is:

```text
06777ebe219da0286be83359565b6179987954c42f2c04f64cd9a8db6dce3bbf
```

Historical terminals, evidence artifacts, denominators, thresholds, role
statuses, and population exposure states are immutable. Later commits do not
supersede or repair them.

## Mutable reporting layer

Reporting-only commits may:

- improve prose and terminology;
- revise figure layout and table formatting;
- move material between the main text and appendices;
- make limitations more prominent;
- narrow claims in response to review;
- add a point-by-point response and publication metadata.

They may not:

- change results, denominators, or thresholds;
- reclassify historical terminals;
- edit historical evidence;
- convert protected 127/128 into qualification;
- rerun Q1 or release protected confirmation;
- introduce new model evidence into this closed lineage.

Every reporting revision after review must identify:

```text
review finding
accepted / partially accepted / disputed
reason
specific reporting change
affected path and section
confirmation that historical evidence is unchanged
```

## External-review boundary

The present agent lineage prepared the evidence and reporting package and
therefore cannot supply the required independent judgment. Review remains
`PENDING` until an uninvolved person returns a digest-bound report.

The reviewer assesses the immutable evidence commit. The provisional paper is
a reporting target and may be revised after findings are received.

