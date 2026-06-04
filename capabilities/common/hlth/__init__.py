"""APG Health Checks and Diagnostics capability.

Standalone package: ``pip install apg-common-hlth``

Quick start::

    from apg_common_hlth import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : hlth
Provides      : health_governance, diagnostic_lifecycle, health_agent_composition, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-hlth"
__capability_id__ = "hlth"

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
