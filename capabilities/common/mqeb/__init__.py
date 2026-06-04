"""APG Message Queue Event Bus capability.

Standalone package: ``pip install apg-common-mqeb``

Quick start::

    from apg_common_mqeb import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mqeb
Provides      : mqeb_event_fabric, message_governance, event_agent_composition, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-mqeb"
__capability_id__ = "mqeb"

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
