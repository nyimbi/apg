"""APG Dispatch Operations capability.

Standalone package: ``pip install apg-transport-dis``

Quick start::

    from apg_transport_dis import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_dis
Provides      : load_planning_workflow, driver_assignment_workflow, dispatch_optimisation_workflow, real_time_tracking_workflow, exception_management_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-dis"
__capability_id__ = "transport_dis"

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
