"""APG Omnichannel Commerce capability.

Standalone package: ``pip install apg-retail-omc``

Quick start::

    from apg_retail_omc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : retail_omc
Provides      : omnichannel_order_management, inventory_visibility, click_and_collect, customer_journey_orchestration, unified_cart, cross_channel_fulfilment
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-retail-omc"
__capability_id__ = "retail_omc"

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
