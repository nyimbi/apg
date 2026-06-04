"""APG Property Contracts capability.

Standalone package: ``pip install apg-realestate-con``

Quick start::

    from apg_realestate_con import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_con
Provides      : contract_lifecycle_management, contractor_registry_management, milestone_tracking_workflow, variation_order_management, dispute_resolution_workflow, contract_clause_library
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-con"
__capability_id__ = "realestate_con"

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
