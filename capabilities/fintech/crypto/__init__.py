"""APG Cryptocurrency Services capability.

Standalone package: ``pip install apg-fintech-crypto``

Quick start::

    from apg_fintech_crypto import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_crypto
Provides      : crypto_asset_workflow, crypto_custody_workflow, crypto_balance_workflow, crypto_order_workflow, crypto_trade_workflow, crypto_transfer_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-crypto"
__capability_id__ = "fintech_crypto"

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
