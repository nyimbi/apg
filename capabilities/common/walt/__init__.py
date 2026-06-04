"""APG Wallet and Payment Core capability.

Standalone package: ``pip install apg-common-walt``

Quick start::

    from apg_common_walt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : walt
Provides      : wallet_ledger, payment_instruments, transaction_authorization, settlement, reconciliation, payment_risk_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-walt"
__capability_id__ = "walt"

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
