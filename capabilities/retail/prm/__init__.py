"""APG Promotions Management capability.

Standalone package: ``pip install apg-retail-prm``

Quick start::

    from apg_retail_prm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : retail_prm
Provides      : promotion_authoring, promotion_activation, pricing_rules_engine, coupon_management, coupon_redemption, markdown_optimisation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-retail-prm"
__capability_id__ = "retail_prm"

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
