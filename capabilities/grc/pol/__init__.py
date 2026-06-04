"""APG Policy Management capability.

Standalone package: ``pip install apg-grc-pol``

Quick start::

    from apg_grc_pol import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : grc_pol
Provides      : policy_lifecycle_management, policy_acknowledgement_workflow, policy_exception_workflow, policy_review_workflow, policy_publication_workflow, policy_dashboard_service
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-grc-pol"
__capability_id__ = "grc_pol"

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
