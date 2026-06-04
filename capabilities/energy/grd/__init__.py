"""APG Grid Operations capability.

Standalone package: ``pip install apg-energy-grd``

Quick start::

    from apg_energy_grd import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : energy_grd
Provides      : real_time_state_estimation, contingency_analysis, voltage_control, frequency_control, market_settlement, grid_alarm_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-energy-grd"
__capability_id__ = "energy_grd"

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
