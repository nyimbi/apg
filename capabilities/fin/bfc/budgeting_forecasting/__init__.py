"""APG Budgeting and Forecasting capability.

Standalone package: ``pip install apg-fin-budgeting_forecasting``

Quick start::

    from apg_fin_budgeting_forecasting import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bfc_budgeting_forecasting
Provides      : budget_planning_lifecycle, budget_line_management, budget_approval_workflow, forecast_lifecycle, scenario_planning, variance_analysis_lifecycle
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-fin-budgeting_forecasting"
__capability_id__ = "bfc_budgeting_forecasting"

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
