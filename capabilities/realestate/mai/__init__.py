"""APG Facilities Maintenance capability.

Standalone package: ``pip install apg-realestate-mai``

Quick start::

    from apg_realestate_mai import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_mai
Provides      : preventive_maintenance_scheduling, work_order_management, contractor_management, asset_lifecycle_tracking, cafm_integration_bridge, sla_monitoring
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-mai"
__capability_id__ = "realestate_mai"

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
