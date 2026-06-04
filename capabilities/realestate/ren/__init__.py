"""APG Rental Operations capability.

Standalone package: ``pip install apg-realestate-ren``

Quick start::

    from apg_realestate_ren import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : realestate_ren
Provides      : tenancy_lifecycle_management, rent_collection_engine, arrears_management_workflow, deposit_accounting, tenancy_renewal_pipeline, referencing_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-realestate-ren"
__capability_id__ = "realestate_ren"

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
