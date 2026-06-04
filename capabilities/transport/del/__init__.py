"""APG Delivery Management capability.

Standalone package: ``pip install apg-transport-del``

Quick start::

    from apg_transport_del import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_del
Provides      : delivery_planning_workflow, proof_of_delivery_workflow, customer_notification_workflow, failed_delivery_workflow, sla_tracking_workflow, delivery_return_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-del"
__capability_id__ = "transport_del"

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
