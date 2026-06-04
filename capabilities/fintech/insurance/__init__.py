"""APG InsurTech capability.

Standalone package: ``pip install apg-fintech-insurance``

Quick start::

    from apg_fintech_insurance import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_insurance
Provides      : insurance_policyholder_workflow, insurance_product_workflow, insurance_quote_workflow, insurance_policy_workflow, insurance_premium_workflow, insurance_claim_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-insurance"
__capability_id__ = "fintech_insurance"

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
