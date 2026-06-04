"""APG Time & Expense Management capability.

Standalone package: ``pip install apg-ppm-tex``

Quick start::

    from apg_ppm_tex import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_tex
Provides      : timesheet_entry_and_management, expense_claim_workflow, approval_workflow_engine, billable_hour_tracking, reimbursement_processing, project_time_reporting
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-tex"
__capability_id__ = "ppm_tex"

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
