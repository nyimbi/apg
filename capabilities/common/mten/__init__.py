"""APG Multi-Tenant Management capability.

Standalone package: ``pip install apg-common-mten``

Quick start::

    from apg_common_mten import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mten
Provides      : mten_operations, tenant_agents, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-mten"
__capability_id__ = "mten"

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
