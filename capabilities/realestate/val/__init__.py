"""APG Property Valuation capability.

Standalone package: ``pip install apg-realestate-val``

Quick start::

    from apg_realestate_val import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_val
Provides      : comparable_sales_analysis, dcf_valuation_engine, mass_appraisal_engine, valuation_roll_management, revaluation_cycle_management, valuation_report_generation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-val"
__capability_id__ = "realestate_val"

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
