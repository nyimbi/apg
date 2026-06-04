"""APG Network Inventory capability.

Standalone package: ``pip install apg-telecom-inv``

Quick start::

    from apg_telecom_inv import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_inv
Provides      : asset_inventory_workflow, circuit_management_workflow, ipam_workflow, topology_documentation_workflow, inventory_reconciliation_workflow, network_resource_query
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-inv"
__capability_id__ = "telecom_inv"

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
