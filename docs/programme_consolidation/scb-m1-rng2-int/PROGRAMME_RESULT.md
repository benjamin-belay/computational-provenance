# Programme result

Status: `MAIN_CSW_AM_SCB_M1_RNG2_INT_FINAL_PROGRAMME_CONSOLIDATED`

## Paper-ready conclusion

In a selected learned model with an explicitly supervised mandatory causal
bottleneck, generated outputs carried a statistical signal derived from an
authenticated abstract internal state. On a prospectively held-out public
population, the signal switched under answer-preserving causal interventions
while the prompt, final answer, semantic payload, and position-indexed random
draws remained fixed. Exact execution provenance, abstract state provenance,
and causal-chain consistency remained separately verifiable.

The public result was 128/128 complete closures. A later protected
qualification did not pass: one of 128 closures failed computation fidelity,
producing a descriptive 127/128 result and two dependent attack-expectation
failures. The frozen adjudicator lacked a scientific-negative terminal and
therefore emitted an implementation-or-evidence-invalid terminal, although
execution chronology, denominators, evidence digests, and independent
reconstruction were valid. Final protected confirmation was not attempted.

## Shortest successful evidence chain

```text
mandatory learned z2 → z3 → answer bottleneck
→ answer-preserving and answer-changing causal transport
→ exact and abstract receipt confirmation
→ position-indexed RNG2 sampler validation
→ calibrated and prospectively validated 16-authority channel
→ prospective public receipt-bound integration validation
```

The programme supports a bounded public proof of concept, not protected
generalisation.

## Central public result

`CSW-AM-SCB-M1-RNG2-INT-V1` retained:

| Measure | Result |
|---|---:|
| Complete closures | 128/128 |
| Natural/alias execution packages | 256 |
| Receipt envelopes | 1,280 |
| Generated outputs | 10,240 |
| Position-indexed model steps | 501,760 |
| Primitive authority scores | 163,840 |
| Aggregate authority scores | 20,480 |
| Frozen attack packages | 1,024 |
| Attack failures | 0 |
| Carrier-off regression events | 0 |
| Wrong-authority confusion events | 0 |

Every closure remained in the denominator. The accepted operating point was:

```text
T = 4.041451884327381
M = 0.0
8 outputs per side
16 candidate authorities
49 position-indexed draws per output
strict score > T
unique rank first
strict margin > M
```

The runtime path verified the abstract receipt and causal chain before
deriving the state authority and authorising generation. The audit path
committed primitive evidence before key reveal and independently reconstructed
receipts, authority derivation, draws, outputs, scores, and verdicts.

## What the construction isolates

For a natural/alias pair, the registered control held fixed:

- visible prompt;
- final answer;
- visible semantic payload;
- generation configuration;
- output indices;
- every position-indexed base random draw.

The intended switching variable was the authenticated abstract `z2` state and
therefore its derived statistical authority. Prompt-, answer-, and
generic-derived authorities remained fixed.

This is stronger than attaching an unauthenticated state label to an output.
The state was architecturally operative, causally intervened upon, and
authenticated before authority derivation.

## Exact and abstract provenance are different claims

The protocol retained separate decisions:

```text
exact execution receipt
abstract state receipt
causal chain
statistical state carrier
```

A different exact execution with the same registered state can therefore
reject exact-execution provenance while accepting abstract-state compatibility
and the corresponding statistical authority. This distinction is part of the
scientific object, not an implementation detail.

## Protected qualification

`CSW-AM-SCB-M1-RNG2-INT-Q1` retained the same full denominators as public V1
and reconstructed the evidence exactly. Its observed result was:

| Measure | Result |
|---|---:|
| Complete closures | 127/128 |
| Failed closure | 1 |
| Computation-fidelity failures | 1 closure |
| Dependent attack failures | 2 |
| Carrier-off regression events | 0 |
| Wrong-authority confusion events | 0 |

This is not a protected pass. The scientific outcome and the terminal-ontology
defect must both be reported:

1. one closure genuinely failed the frozen computation-fidelity gate; and
2. the positive-only adjudicator could not encode that valid scientific
   negative and emitted
   `MAIN_CSW_AM_SCB_M1_RNG2_INT_Q1_IMPLEMENTATION_OR_EVIDENCE_INVALID`.

The terminal must not be relabelled post hoc. The 256-closure protected
confirmation population remains sealed and unused.

## Sidecar comparison

A sidecar or inline HMAC can authenticate a declared state with stronger
cryptographic integrity and less statistical uncertainty. This programme
demonstrates a different property: an already authenticated causal state
controls a statistical signal expressed in generated output. It does not
claim that the statistical carrier is a superior replacement for an HMAC.

## Contribution framing

The strongest defensible framing is:

> A feasibility and architecture result for authenticated causal-state
> carriage in generated output under a bounded, trusted, engineered system.

It is not a general solution to model-internal provenance.

