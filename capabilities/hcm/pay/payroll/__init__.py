"""APG Payroll capability.

Standalone package: ``pip install apg-hcm-payroll``

Quick start::

    from apg_hcm_payroll import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pay_payroll
Provides      : payroll_period_lifecycle, pay_group_lifecycle, employee_pay_profile_lifecycle, pay_component_lifecycle, time_import_lifecycle, payroll_run_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-hcm-payroll"
__capability_id__ = "pay_payroll"

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
