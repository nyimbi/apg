"""APG Know Your Customer capability.

Standalone package: ``pip install apg-fintech-kyc``

Quick start::

    from apg_fintech_kyc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_kyc
Provides      : customer_identity_lifecycle, document_verification_workflow, sanctions_pep_screening, kyc_risk_scoring, customer_due_diligence, enhanced_due_diligence
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-kyc"
__capability_id__ = "fintech_kyc"

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
