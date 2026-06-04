"""APG Real-Time Collaboration capability.

Standalone package: ``pip install apg-ckm-rtc``

Quick start::

    from apg_ckm_rtc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ckm_rtc
Provides      : collaboration_sessions, presence_awareness, real_time_messaging, media_collaboration, decision_capture, page_collaboration
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ckm-rtc"
__capability_id__ = "ckm_rtc"

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
