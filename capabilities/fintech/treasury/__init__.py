"""APG Treasury Management System capability.

Standalone package: ``pip install apg-fintech-treasury``

Quick start::

    from apg_fintech_treasury import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_treasury
Provides      : cash_position_management, treasury_dealing_workflow, counterparty_limit_governance, settlement_instruction_workflow, fx_rate_management, liquidity_forecasting
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-treasury"
__capability_id__ = "fintech_treasury"

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
