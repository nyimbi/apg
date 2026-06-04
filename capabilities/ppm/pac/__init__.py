"""APG Project Accounting capability.

Standalone package: ``pip install apg-ppm-pac``

Quick start::

    from apg_ppm_pac import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_pac
Provides      : project_cost_tracking, revenue_recognition_workflow, wip_accounting_workflow, milestone_billing_workflow, project_profitability_reporting, budget_vs_actual_analysis
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-pac"
__capability_id__ = "ppm_pac"

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
