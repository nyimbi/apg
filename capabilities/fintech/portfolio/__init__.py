"""APG Portfolio Management capability.

Standalone package: ``pip install apg-fintech-portfolio``

Quick start::

    from apg_fintech_portfolio import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_portfolio
Provides      : portfolio_book_workflow, portfolio_holding_workflow, portfolio_allocation_policy_workflow, portfolio_valuation_workflow, portfolio_benchmark_workflow, portfolio_risk_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-portfolio"
__capability_id__ = "fintech_portfolio"

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
