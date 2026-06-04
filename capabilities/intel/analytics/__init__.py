"""APG Intelligence Analytics capability.

Standalone package: ``pip install apg-intel-analytics``

Quick start::

    from apg_intel_analytics import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_analytics
Provides      : analytics_authority_workflow, analytics_workspace_workflow, analytics_dataset_workflow, analytics_feature_workflow, analytics_model_workflow, analytics_run_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-analytics"
__capability_id__ = "intel_analytics"

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
