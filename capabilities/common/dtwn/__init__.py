"""APG Digital Twin Framework capability.

Standalone package: ``pip install apg-common-dtwn``

Quick start::

    from apg_common_dtwn import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : dtwn
Provides      : twin_registry, simulation_models, telemetry_fusion, prediction_workflows, asset_topology, twin_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-dtwn"
__capability_id__ = "dtwn"

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
