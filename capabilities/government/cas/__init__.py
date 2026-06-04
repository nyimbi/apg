"""APG Case Management capability.

Standalone package: ``pip install apg-government-cas``

Quick start::

    from apg_government_cas import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_cas
Provides      : case_intake_workflow, case_assignment_workflow, case_routing_workflow, sla_tracking_workflow, case_escalation_workflow, case_outcome_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-cas"
__capability_id__ = "government_cas"

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
