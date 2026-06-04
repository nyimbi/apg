"""APG Space Planning & Management capability.

Standalone package: ``pip install apg-realestate-spa``

Quick start::

    from apg_realestate_spa import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_spa
Provides      : floor_plan_management, space_allocation_engine, move_management_workflow, occupancy_analytics, workplace_density_planning, space_booking_engine
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-spa"
__capability_id__ = "realestate_spa"

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
