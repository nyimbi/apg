"""APG Digital Wallets capability.

Standalone package: ``pip install apg-fintech-wallets``

Quick start::

    from apg_fintech_wallets import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_wallets
Provides      : wallet_lifecycle, stored_value_ledger, wallet_instrument_registry, wallet_transfer_workflow, wallet_hold_workflow, wallet_limit_governance
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-wallets"
__capability_id__ = "fintech_wallets"

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
