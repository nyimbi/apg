"""APG Learning Management System capability.

Standalone package: ``pip install apg-education-lms``

Quick start::

    from apg_education_lms import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : education_lms
Provides      : course_lifecycle_workflow, content_delivery_workflow, enrolment_workflow, assessment_workflow, grading_workflow, certificate_issuance_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-education-lms"
__capability_id__ = "education_lms"

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
