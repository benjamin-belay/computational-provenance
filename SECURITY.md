# Security and disclosure boundary

This release intentionally excludes private or unrevealed cryptographic keys,
controller secrets, sealed protected identities, and controller workspaces.

Some frozen configurations retain absolute paths from the original controlled
environment as provenance records. Those paths do not contain credentials and
will not exist on another machine. Do not replace them in-place when auditing
the frozen source; create a separate replay configuration instead.

Please report accidental disclosure or integrity concerns privately to the
repository owner before opening a public issue.
