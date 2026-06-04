"""APG Tax Administration capability.

Standalone package: ``pip install apg-government-tax``

Quick start::

    from apg_government_tax import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_tax
Provides      : taxpayer_registration_workflow, return_filing_workflow, tax_assessment_workflow, objection_management_workflow, debt_collection_workflow, audit_case_management_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-tax"
__capability_id__ = "government_tax"

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
