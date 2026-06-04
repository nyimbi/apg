"""APG Multi-Currency Management capability.

Standalone package: ``pip install apg-loc-mcy``

Quick start::

    from apg_loc_mcy import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : loc_mcy
Provides      : currency_configuration, exchange_rate_management, fx_revaluation_workflow, currency_translation_workflow, fx_gain_loss_reporting, multi_currency_rounding
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-loc-mcy"
__capability_id__ = "loc_mcy"

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
