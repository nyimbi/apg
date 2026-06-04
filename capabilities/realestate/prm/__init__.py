"""APG Property Management capability.

Standalone package: ``pip install apg-realestate-prm``

Quick start::

    from apg_realestate_prm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_prm
Provides      : property_portfolio_management, unit_management, owner_portal_service, property_performance_reporting, portfolio_analytics, handover_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-prm"
__capability_id__ = "realestate_prm"

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
