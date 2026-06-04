"""APG Real Estate Accounting capability.

Standalone package: ``pip install apg-realestate-acc``

Quick start::

    from apg_realestate_acc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_acc
Provides      : property_ledger_management, service_charge_accounting, cam_reconciliation_workflow, ifrs16_lease_accounting, revenue_recognition_engine, journal_entry_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-acc"
__capability_id__ = "realestate_acc"

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
