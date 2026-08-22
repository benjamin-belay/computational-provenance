# Security and trust model

The receipt mechanism uses symmetric HMAC authentication. Verifiers therefore trust the receipt issuer and must possess the verification key. This is not a public-key attestation system.

No original secret receipt or sampler keys are included. The replay derives local deterministic keys solely for reproducibility. Do not reuse them outside the reproduction run.

Please report accidental credential exposure privately to the repository owner rather than opening a public issue.
