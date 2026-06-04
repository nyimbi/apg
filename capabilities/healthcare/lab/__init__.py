"""APG Laboratory Information System capability.

Standalone package: ``pip install apg-healthcare-lab``

Quick start::

    from apg_healthcare_lab import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_lab
Provides      : lab_order_management, specimen_tracking, result_entry_verification, critical_value_alerting, qc_management, instrument_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-lab"
__capability_id__ = "healthcare_lab"

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
