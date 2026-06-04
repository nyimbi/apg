"""APG Access Control Integration Hub capability.

Standalone package: ``pip install apg-composition-access``

Quick start::

    from apg_composition_access import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_access
Provides      : identity_provider_composition, resource_access_registry, policy_orchestration, grant_lifecycle, session_risk_control, access_decision_audit
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-access"
__capability_id__ = "composition_access"

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
