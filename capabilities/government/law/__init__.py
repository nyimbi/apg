"""APG Law Enforcement and Justice capability.

Standalone package: ``pip install apg-government-law``

Quick start::

    from apg_government_law import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_law
Provides      : incident_reporting_workflow, docket_management_workflow, evidence_chain_of_custody_workflow, court_scheduling_workflow, prosecution_tracking_workflow, law_enforcement_review_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-law"
__capability_id__ = "government_law"

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
