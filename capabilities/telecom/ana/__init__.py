"""APG Telecom Analytics capability.

Standalone package: ``pip install apg-telecom-ana``

Quick start::

    from apg_telecom_ana import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_ana
Provides      : analytics_pipeline, churn_prediction_workflow, arpu_analysis_workflow, usage_pattern_workflow, revenue_assurance_workflow, network_performance_analytics
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-ana"
__capability_id__ = "telecom_ana"

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
