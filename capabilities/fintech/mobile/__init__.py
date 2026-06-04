"""APG Mobile Banking capability.

Standalone package: ``pip install apg-fintech-mobile``

Quick start::

    from apg_fintech_mobile import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_mobile
Provides      : mobile_banking_program_governance, mobile_customer_enrollment, trusted_device_lifecycle, mobile_authentication_factor_workflow, mobile_account_linking, mobile_payment_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-mobile"
__capability_id__ = "fintech_mobile"

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
