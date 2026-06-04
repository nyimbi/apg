"""APG API Service Mesh capability.

Standalone package: ``pip install apg-composition-gateway``

Quick start::

    from apg_composition_gateway import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_gateway
Provides      : service_mesh_registry, gateway_route_lifecycle, traffic_management, gateway_policy_enforcement, certificate_lifecycle, mesh_health_observability
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-gateway"
__capability_id__ = "composition_gateway"

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
