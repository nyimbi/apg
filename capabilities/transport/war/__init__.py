"""APG Warehouse Operations capability.

Standalone package: ``pip install apg-transport-war``

Quick start::

    from apg_transport_war import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_war
Provides      : warehouse_receiving_workflow, putaway_workflow, picking_workflow, packing_workflow, cross_docking_workflow, cycle_counting_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-war"
__capability_id__ = "transport_war"

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
