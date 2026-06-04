"""APG Digital Lending capability.

Standalone package: ``pip install apg-fintech-lending``

Quick start::

    from apg_fintech_lending import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_lending
Provides      : loan_product_governance, borrower_lifecycle, credit_application_workflow, underwriting_decisioning, loan_offer_workflow, disbursement_control
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-lending"
__capability_id__ = "fintech_lending"

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
