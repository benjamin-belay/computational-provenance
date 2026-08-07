# Related-work positioning notes

These notes freeze the intended comparisons for the provisional paper. They
do not claim an exhaustive literature review.

| Area | Prior object | Relationship to this programme |
|---|---|---|
| Statistical LLM watermarking | Detect that text was produced under a keyed sampling rule | Reuses the general idea of statistical token carriage, but the authority denotes an authenticated internal state class |
| SynthID-Text | Production-oriented text watermarking through sampling modification | “SynthID-style” here means only statistical output carriage; no equivalence or compatibility claim |
| Causal abstraction | Align neural representations with causal variables and test counterfactual behavior | Supplies the causal interpretation; this programme uses an explicitly supervised mandatory bottleneck |
| Interchange-intervention training | Induce a registered causal abstraction during learning | Closest architectural precedent; this programme adds receipts and output carriage |
| Content credentials / C2PA | Signed assertions and hard/soft bindings to assets | Conceptual provenance comparison only; this system uses trusted HMAC receipts and is not C2PA |
| Sidecar HMAC | Cryptographically authenticate declared metadata | Simpler and stronger for metadata integrity; does not by itself express the state as a statistical property of output |

## Primary sources

- Dathathri et al. (2024),
  [Scalable watermarking for identifying large language model
  outputs](https://www.nature.com/articles/s41586-024-08025-4).
- Kirchenbauer et al. (2023),
  [A Watermark for Large Language
  Models](https://proceedings.mlr.press/v202/kirchenbauer23a.html).
- Geiger et al. (2021),
  [Causal Abstractions of Neural
  Networks](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html).
- Geiger et al. (2022),
  [Inducing Causal Structure for Interpretable Neural
  Networks](https://proceedings.mlr.press/v162/geiger22a.html).
- Coalition for Content Provenance and Authenticity (2025),
  [C2PA Technical Specification
  2.2](https://spec.c2pa.org/specifications/specifications/2.2/specs/C2PA_Specification.html).

## Wording to avoid

- “SynthID for reasoning”
- “cryptographic watermark”
- “the output proves the exact execution”
- “general mechanistic provenance”
- “protected replication succeeded”
- “state discovered in an unconstrained model”

