"""APG Portfolio Analytics capability.

Standalone package: ``pip install apg-ppm-pan``

Quick start::

    from apg_ppm_pan import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_pan
Provides      : portfolio_performance_dashboard, strategic_alignment_scoring, risk_return_analysis, capacity_heat_map, portfolio_investment_analysis, project_pipeline_reporting
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-pan"
__capability_id__ = "ppm_pan"

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
