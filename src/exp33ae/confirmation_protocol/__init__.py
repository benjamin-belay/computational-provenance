"""Prospectively frozen 33a-E confirmation protocol.

Importing this package does not read entropy, allocate data, create output, or
authorize confirmation.  Claim-bearing launch remains guarded by external
objects that are deliberately absent from this repository.
"""

from .normative import PROTOCOL_VERSION, protocol_hash

__all__ = ["PROTOCOL_VERSION", "protocol_hash"]
