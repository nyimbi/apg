"""APG Electoral and Civil Registration capability.

Standalone package: ``pip install apg-government-ele``

Quick start::

    from apg_government_ele import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_ele
Provides      : voter_registration_workflow, biometric_deduplication_workflow, polling_station_management_workflow, election_management_workflow, results_collation_workflow, civil_registration_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-ele"
__capability_id__ = "government_ele"

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
