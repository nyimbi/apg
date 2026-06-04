"""APG Licensing and Permits capability.

Standalone package: ``pip install apg-government-lic``

Quick start::

    from apg_government_lic import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_lic
Provides      : licence_application_workflow, licence_issuance_workflow, inspection_scheduling_workflow, licence_renewal_workflow, fee_collection_workflow, licence_revocation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-lic"
__capability_id__ = "government_lic"

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
