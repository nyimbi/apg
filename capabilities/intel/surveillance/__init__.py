"""APG Digital Surveillance capability.

Standalone package: ``pip install apg-intel-surveillance``

Quick start::

    from apg_intel_surveillance import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_surveillance
Provides      : surveillance_authority_workflow, surveillance_program_workflow, surveillance_asset_workflow, surveillance_sensor_workflow, surveillance_observation_workflow, surveillance_alert_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-surveillance"
__capability_id__ = "intel_surveillance"

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
