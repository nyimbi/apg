"""APG Fuel Management capability.

Standalone package: ``pip install apg-transport-fue``

Quick start::

    from apg_transport_fue import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_fue
Provides      : fuel_procurement_workflow, fuel_consumption_tracking_workflow, bunker_management_workflow, fuel_card_reconciliation_workflow, carbon_footprint_reporting_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-fue"
__capability_id__ = "transport_fue"

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
