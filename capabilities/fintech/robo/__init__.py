"""APG Robo Advisory capability.

Standalone package: ``pip install apg-fintech-robo``

Quick start::

    from apg_fintech_robo import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_robo
Provides      : robo_investor_profile_workflow, robo_goal_plan_workflow, robo_model_portfolio_workflow, robo_recommendation_workflow, robo_automation_workflow, robo_drift_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-robo"
__capability_id__ = "fintech_robo"

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
