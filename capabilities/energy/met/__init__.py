"""APG Smart Metering & AMI capability.

Standalone package: ``pip install apg-energy-met``

Quick start::

    from apg_energy_met import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : energy_met
Provides      : meter_registry, ami_head_end_management, interval_data_collection, tamper_detection, remote_connect_disconnect, demand_response_coordination
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-energy-met"
__capability_id__ = "energy_met"

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
