"""APG Pharmacy Management capability.

Standalone package: ``pip install apg-healthcare-pha``

Quick start::

    from apg_healthcare_pha import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_pha
Provides      : drug_formulary_management, prescription_dispensing, lasa_alert_management, controlled_substance_tracking, drug_interaction_checking, pharmacy_inventory_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-pha"
__capability_id__ = "healthcare_pha"

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
