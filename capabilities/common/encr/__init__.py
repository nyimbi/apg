"""APG Encryption Services capability.

Standalone package: ``pip install apg-common-encr``

Quick start::

    from apg_common_encr import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : encr
Provides      : encr_operations, crypto_governance, crypto_agent_composition, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-encr"
__capability_id__ = "encr"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
]

# Backward-compatibility stub

class APGEncryptionInterface:
    @staticmethod
    def encrypt(data: bytes, key: bytes) -> bytes: return data
    @staticmethod
    def decrypt(data: bytes, key: bytes) -> bytes: return data

# Module-level interface instance
class _EncryptionInterface:
    @staticmethod
    def encrypt(data: bytes, key_id: str = "") -> bytes: return data
    @staticmethod
    def decrypt(data: bytes, key_id: str = "") -> bytes: return data

encryption_interface = _EncryptionInterface()
