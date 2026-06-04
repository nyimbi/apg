"""APG Notification System capability.

Standalone package: ``pip install apg-ckm-not``

Quick start::

    from apg_ckm_not import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ckm_not
Provides      : notification_delivery, template_management, campaign_orchestration, preference_center, channel_provider_registry, engagement_analytics
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ckm-not"
__capability_id__ = "ckm_not"

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
