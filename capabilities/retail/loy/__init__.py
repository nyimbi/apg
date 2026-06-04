"""APG Loyalty & Rewards capability.

Standalone package: ``pip install apg-retail-loy``

Quick start::

    from apg_retail_loy import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : retail_loy
Provides      : loyalty_member_enrolment, loyalty_points_earn, loyalty_points_redeem, loyalty_tier_management, loyalty_campaign_management, loyalty_partner_coalition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-retail-loy"
__capability_id__ = "retail_loy"

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
