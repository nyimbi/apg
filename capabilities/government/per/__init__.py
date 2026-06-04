"""APG Permits Management capability.

Standalone package: ``pip install apg-government-per``

Quick start::

    from apg_government_per import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_per
Provides      : permit_application_workflow, permit_issuance_workflow, conditional_approval_workflow, inspection_scheduling_workflow, permit_compliance_monitoring_workflow, permit_revocation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-per"
__capability_id__ = "government_per"

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
