"""APG fintech_gateway capability.

Standalone package: ``pip install apg-fintech-gateway``

Quick start::

    from apg_fintech_gateway import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_gateway
Provides      : merchant_onboarding_lifecycle, provider_connection_lifecycle, payment_method_tokenization_workflow, payment_intent_lifecycle, payment_routing_workflow, fraud_risk_review_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-fintech-gateway"
__capability_id__ = "fintech_gateway"

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
