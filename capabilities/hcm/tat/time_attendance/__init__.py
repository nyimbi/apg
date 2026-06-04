"""APG Time and Attendance capability.

Standalone package: ``pip install apg-hcm-time_attendance``

Quick start::

    from apg_hcm_time_attendance import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : tat_time_attendance
Provides      : time_policy_lifecycle, work_schedule_lifecycle, shift_lifecycle, time_entry_lifecycle, break_lifecycle, timesheet_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-hcm-time_attendance"
__capability_id__ = "tat_time_attendance"

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
