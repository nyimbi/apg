"""APG Budget Management capability.

Standalone package: ``pip install apg-government-bud``

Quick start::

    from apg_government_bud import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_bud
Provides      : budget_programme_workflow, vote_accounting_workflow, budget_revision_workflow, commitment_control_workflow, expenditure_recording_workflow, fiscal_reporting_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-bud"
__capability_id__ = "government_bud"

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
