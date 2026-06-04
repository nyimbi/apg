"""APG IoT Device Integration capability.

Standalone package: ``pip install apg-common-iotd``

Quick start::

    from apg_common_iotd import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : iotd
Provides      : device_registry, telemetry_ingestion, command_dispatch, firmware_lifecycle, device_security, device_health
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-iotd"
__capability_id__ = "iotd"

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
