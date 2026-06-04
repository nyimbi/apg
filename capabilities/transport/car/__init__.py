"""APG Cargo Management capability.

Standalone package: ``pip install apg-transport-car``

Quick start::

    from apg_transport_car import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_car
Provides      : cargo_booking_workflow, cargo_manifest_workflow, dangerous_goods_compliance_workflow, cargo_tracking_workflow, cargo_revenue_workflow, cargo_compliance_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-car"
__capability_id__ = "transport_car"

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
