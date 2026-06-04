"""APG Asset Tracking capability.

Standalone package: ``pip install apg-transport-tra``

Quick start::

    from apg_transport_tra import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : transport_tra
Provides      : realtime_gps_tracking_workflow, geofencing_workflow, cold_chain_monitoring_workflow, container_tracking_workflow, asset_utilisation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-transport-tra"
__capability_id__ = "transport_tra"

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
