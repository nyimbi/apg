"""APG Multi-Country Operations capability.

Standalone package: ``pip install apg-loc-mco``

Quick start::

    from apg_loc_mco import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : loc_mco
Provides      : country_entity_management, regulatory_compliance_mapping, intercompany_transaction_workflow, statutory_reporting_workflow, transfer_pricing_validation, cross_border_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-loc-mco"
__capability_id__ = "loc_mco"

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
