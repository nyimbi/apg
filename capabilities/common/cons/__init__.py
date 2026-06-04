"""APG Consent and Privacy Management capability.

Standalone package: ``pip install apg-common-cons``

Quick start::

    from apg_common_cons import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : cons
Provides      : purpose_registry, consent_capture, privacy_requests, preference_center, privacy_audit, privacy_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-cons"
__capability_id__ = "cons"

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
