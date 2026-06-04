"""APG Integration API Management capability.

Standalone package: ``pip install apg-int-api``

Quick start::

    from apg_int_api import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : int_api
Provides      : api_registry_lifecycle, api_endpoint_lifecycle, api_policy_lifecycle, api_consumer_lifecycle, api_key_lifecycle, api_subscription_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-int-api"
__capability_id__ = "int_api"

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
