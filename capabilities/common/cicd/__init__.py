"""APG Continuous Integration and Delivery capability.

Standalone package: ``pip install apg-common-cicd``

Quick start::

    from apg_common_cicd import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : cicd
Provides      : pipeline_management, build_orchestration, quality_gates, artifact_promotion, release_automation, delivery_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-cicd"
__capability_id__ = "cicd"

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
