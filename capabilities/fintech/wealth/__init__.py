"""APG Wealth Management capability.

Standalone package: ``pip install apg-fintech-wealth``

Quick start::

    from apg_fintech_wealth import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_wealth
Provides      : wealth_client_profile_workflow, suitability_profile_workflow, portfolio_management_workflow, advisory_mandate_workflow, portfolio_rebalance_workflow, wealth_order_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-wealth"
__capability_id__ = "fintech_wealth"

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
