"""Structurally DEV_ONLY preconfirmation protocol infrastructure.

No function in this package can instantiate a claim-bearing confirmation stage.
"""

from .normative import PROTOCOL_RELATIVE_PATH, protocol_hash, load_protocol

__all__ = ["PROTOCOL_RELATIVE_PATH", "protocol_hash", "load_protocol"]
