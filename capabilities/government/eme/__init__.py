"""APG Emergency Management capability.

Standalone package: ``pip install apg-government-eme``

Quick start::

    from apg_government_eme import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_eme
Provides      : incident_command_workflow, resource_mobilisation_workflow, multi_agency_coordination_workflow, eoc_management_workflow, situation_reporting_workflow, after_action_review_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-eme"
__capability_id__ = "government_eme"

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
