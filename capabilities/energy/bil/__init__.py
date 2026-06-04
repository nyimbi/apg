"""APG Energy Billing & Tariffs capability.

Standalone package: ``pip install apg-energy-bil``

Quick start::

    from apg_energy_bil import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : energy_bil
Provides      : tariff_management, consumption_billing, demand_charge_calculation, renewable_credits_management, revenue_assurance, payment_processing
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-energy-bil"
__capability_id__ = "energy_bil"

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
