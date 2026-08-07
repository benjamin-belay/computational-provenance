# Environment and hardware record

This records the audit/development host visible when the archival handoff was
created. It is not a claim that every historical execution used this exact
host.

## Locked project dependencies

The repository declares:

```text
Python >=3.12,<3.13
torch==2.7.1
numpy==2.2.6
pytest==8.4.1
```

The handoff host reported:

```text
Python 3.12.13
PyTorch 2.7.1
NumPy 2.2.6
SciPy 1.16.1
scikit-learn 1.7.1
macOS 26.2 (build 25C56)
Darwin 25.2.0, arm64
Apple M3 Pro, 11 CPU cores, 18 GB memory
MacBookPro15,6
```

Dependency snapshots from the source research repository are retained at
`environment/pyproject.toml` and `environment/requirements-lock.txt`.

Hardware serial numbers, UUIDs, and device identifiers are intentionally
excluded from this public record.

## Determinism boundary

The public retained-result audit does not require a model forward. Exact
regeneration is more environment-sensitive and should use the frozen
dependency versions, source commit, keys, role manifests, and commands stored
with each stage. Re-execution never replaces the retained prospective
evidence.
