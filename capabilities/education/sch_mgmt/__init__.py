"""APG School Management capability.

Standalone package: ``pip install apg-education-sch_mgmt``

Quick start::

    from apg_education_sch_mgmt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : education_sch_mgmt
Provides      : student_records_workflow, admissions_workflow, fee_management_workflow, parent_portal_workflow, staff_administration_workflow, academic_calendar_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-education-sch_mgmt"
__capability_id__ = "education_sch_mgmt"

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
