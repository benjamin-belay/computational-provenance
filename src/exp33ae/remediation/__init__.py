"""Strictly DEV_ONLY 33a-E remediation runner.

The package has no enabled claim-bearing stage. A non-DEV request is rejected
before entropy is accepted or derived.
"""

from .normative import DEV_NAMESPACE, POSITIVE_ANALOGUE, protocol_hash

__all__ = ["DEV_NAMESPACE", "POSITIVE_ANALOGUE", "protocol_hash"]
