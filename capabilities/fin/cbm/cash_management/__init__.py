"""APG cbm_cash_management capability.

Standalone package: ``pip install apg-fin-cash_management``

Quick start::

    from apg_fin_cash_management import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : cbm_cash_management
Provides      : bank_relationship_lifecycle, cash_account_lifecycle, cash_position_service, cash_flow_lifecycle, cash_forecasting_workflow, liquidity_control_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-fin-cash_management"
__capability_id__ = "cbm_cash_management"

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
