"""APG Employee Data Management capability.

Standalone package: ``pip install apg-hcm-employee_data_management``

Quick start::

    from apg_hcm_employee_data_management import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : chr_employee_data_management
Provides      : employee_profile_lifecycle, employee_identity_registry, department_lifecycle, position_lifecycle, employment_history_lifecycle, employee_skill_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-hcm-employee_data_management"
__capability_id__ = "chr_employee_data_management"

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
