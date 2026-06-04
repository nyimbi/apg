"""APG Project Baseline Management capability.

Standalone package: ``pip install apg-ppm-pbl``

Quick start::

    from apg_ppm_pbl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_pbl
Provides      : scope_baseline_management, schedule_baseline_management, cost_baseline_management, change_control_workflow, earned_value_analysis, baseline_variance_tracking
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-pbl"
__capability_id__ = "ppm_pbl"

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
