"""APG Resource Management capability.

Standalone package: ``pip install apg-ppm-res``

Quick start::

    from apg_ppm_res import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_res
Provides      : resource_pool_management, skill_matching_engine, capacity_planning, utilisation_tracking, demand_forecasting, resource_allocation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-res"
__capability_id__ = "ppm_res"

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
