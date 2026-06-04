"""APG ESG and Carbon Tracking capability.

Standalone package: ``pip install apg-common-esgc``

Quick start::

    from apg_common_esgc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : esgc
Provides      : emissions_inventory, factor_library, activity_emissions, sustainability_reporting, target_tracking, esg_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-esgc"
__capability_id__ = "esgc"

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
