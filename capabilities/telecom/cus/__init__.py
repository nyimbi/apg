"""APG Customer Management capability.

Standalone package: ``pip install apg-telecom-cus``

Quick start::

    from apg_telecom_cus import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_cus
Provides      : customer_lifecycle_workflow, kyc_workflow, plan_management_workflow, sim_management_workflow, device_management_workflow, case_tracking_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-cus"
__capability_id__ = "telecom_cus"

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
