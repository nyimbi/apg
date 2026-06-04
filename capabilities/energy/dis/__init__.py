"""APG Distribution Network capability.

Standalone package: ``pip install apg-energy-dis``

Quick start::

    from apg_energy_dis import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : energy_dis
Provides      : network_topology_management, fault_detection_and_isolation, outage_restoration, switching_order_management, scada_integration, load_balancing
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-energy-dis"
__capability_id__ = "energy_dis"

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
