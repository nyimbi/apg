"""APG Route Optimisation capability.

Standalone package: ``pip install apg-transport-rou``

Quick start::

    from apg_transport_rou import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_rou
Provides      : multi_stop_route_planning_workflow, dynamic_rerouting_workflow, traffic_integration_workflow, time_window_constraint_workflow, multimodal_routing_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-rou"
__capability_id__ = "transport_rou"

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
