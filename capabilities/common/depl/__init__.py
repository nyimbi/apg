"""APG Deployment Management capability.

Standalone package: ``pip install apg-common-depl``

Quick start::

    from apg_common_depl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : depl
Provides      : release_management, deployment_rollouts, health_gates, rollback_control, deployment_audit, deployment_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-depl"
__capability_id__ = "depl"

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
