"""APG Vendor Management capability.

Standalone package: ``pip install apg-scm-ven``

Quick start::

    from apg_scm_ven import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : scm_ven
Provides      : vendor_profile_lifecycle, vendor_onboarding_workflow, vendor_qualification_lifecycle, vendor_performance_lifecycle, vendor_risk_lifecycle, vendor_contract_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-scm-ven"
__capability_id__ = "scm_ven"

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
