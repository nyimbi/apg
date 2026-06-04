"""APG Renewable Energy capability.

Standalone package: ``pip install apg-energy-ren``

Quick start::

    from apg_energy_ren import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : energy_ren
Provides      : renewable_asset_registry, curtailment_tracking, rec_certificate_management, carbon_credit_management, feed_in_tariff_management, generation_forecasting
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-energy-ren"
__capability_id__ = "energy_ren"

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
