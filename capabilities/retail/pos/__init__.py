"""APG Point of Sale capability.

Standalone package: ``pip install apg-retail-pos``

Quick start::

    from apg_retail_pos import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : retail_pos
Provides      : pos_transaction_processing, pos_session_management, pos_cash_management, pos_till_reconciliation, pos_receipt_management, pos_discount_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-retail-pos"
__capability_id__ = "retail_pos"

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
