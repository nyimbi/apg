"""APG Timetabling & Scheduling capability.

Standalone package: ``pip install apg-education-ttbl``

Quick start::

    from apg_education_ttbl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : education_ttbl
Provides      : timetable_generation_workflow, constraint_management_workflow, room_allocation_workflow, teacher_assignment_workflow, conflict_detection_workflow, conflict_resolution_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-education-ttbl"
__capability_id__ = "education_ttbl"

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
