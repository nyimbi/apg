"""APG Medical Device Management capability.

Standalone package: ``pip install apg-healthcare-dev``

Quick start::

    from apg_healthcare_dev import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_dev
Provides      : device_inventory_management, maintenance_schedule_management, calibration_record_tracking, fda_udi_tracking, adverse_event_reporting, work_order_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-dev"
__capability_id__ = "healthcare_dev"

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
