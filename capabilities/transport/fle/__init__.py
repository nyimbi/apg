"""APG Fleet Management capability.

Standalone package: ``pip install apg-transport-fle``

Quick start::

    from apg_transport_fle import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_fle
Provides      : vehicle_lifecycle_workflow, telematics_integration_workflow, driver_management_workflow, fleet_utilisation_analytics_workflow, fleet_compliance_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-fle"
__capability_id__ = "transport_fle"

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
