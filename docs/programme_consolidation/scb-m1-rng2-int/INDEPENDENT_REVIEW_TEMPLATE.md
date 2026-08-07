# Independent methodological review

Reviewer:

Affiliation or declared independence:

Review date:

Repository commit:

Handoff manifest SHA-256:

## Required review questions

For each question, record `YES`, `NO`, or `UNCLEAR`, cite the exact artifact or
source location, and explain any qualification.

1. Does INT-V1 isolate internal computation from prompt, answer, semantic
   payload, and primitive randomness?
2. Is the authenticated internal state causally operative rather than merely
   labelled?
3. Is statistical authority derived only after abstract-receipt and
   causal-chain verification?
4. Are exact-execution provenance and abstract-state provenance genuinely
   distinct in schemas, controls, and verdicts?
5. Are public validation and protected qualification reported separately?
6. Does any headline, abstract, figure, or conclusion imply that protected
   confirmation occurred or that protected qualification passed?
7. Is “SynthID-style” clearly limited to statistical output carriage, without
   implying equivalence to a production SynthID system?
8. Are the engineered-state, selected-model, trusted-controller, and
   trusted-HMAC assumptions prominent enough?
9. Could a simpler sidecar or inline HMAC explain the claimed advantage, and
   is the comparison framed as output carriage rather than superior
   cryptographic integrity?
10. Is the contribution best described as bounded feasibility and
    architecture, rather than general provenance?

## Reproduction checks

- [ ] `verify_handoff.py` returns the expected terminal.
- [ ] All manifest evidence digests match.
- [ ] Public INT-V1 reconstructs 128/128 and zero frozen attack regressions.
- [ ] Q1 reconstructs 127/128, one computation failure, and two dependent
      attack failures.
- [ ] The final terminal remains unchanged.
- [ ] The protected confirmation root and zero-use attestation agree.
- [ ] No protected confirmation identity is present in the handoff.

## Reviewer conclusion

Overall assessment:

Required claim changes:

Required reproducibility changes:

Unresolved methodological concerns:

Signature or signed-report digest:

