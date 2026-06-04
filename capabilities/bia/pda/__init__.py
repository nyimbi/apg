"""APG Predictive Analytics capability.

Standalone package: ``pip install apg-bia-pda``

Quick start::

    from apg_bia_pda import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_pda
Provides      : ml_model_training, demand_forecasting, trend_analysis, regression_modelling, scenario_simulation, anomaly_prediction
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-pda"
__capability_id__ = "bia_pda"

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
