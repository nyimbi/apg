"""APG Algorithmic Trading capability.

Standalone package: ``pip install apg-fintech-trading``

Quick start::

    from apg_fintech_trading import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_trading
Provides      : trading_strategy_workflow, trading_signal_workflow, trading_backtest_workflow, trading_risk_limit_workflow, trading_order_intent_workflow, trading_execution_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-trading"
__capability_id__ = "fintech_trading"

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
