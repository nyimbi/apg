"""APG Banking APIs capability.

Standalone package: ``pip install apg-fintech-apis``

Quick start::

    from apg_fintech_apis import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_apis
Provides      : banking_api_product_governance, developer_onboarding_workflow, developer_application_workflow, banking_consent_workflow, api_client_credential_workflow, api_endpoint_policy_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-apis"
__capability_id__ = "fintech_apis"

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
