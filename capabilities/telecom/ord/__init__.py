"""APG Order Management capability.

Standalone package: ``pip install apg-telecom-ord``

Quick start::

    from apg_telecom_ord import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_ord
Provides      : order_capture_workflow, order_validation_workflow, order_decomposition_workflow, provisioning_orchestration_workflow, fallout_management_workflow, order_tracking_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-ord"
__capability_id__ = "telecom_ord"

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
