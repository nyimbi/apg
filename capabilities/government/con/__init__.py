"""APG Government Contracts and Procurement capability.

Standalone package: ``pip install apg-government-con``

Quick start::

    from apg_government_con import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_con
Provides      : tender_management_workflow, evaluation_workflow, contract_award_workflow, contract_lifecycle_workflow, contract_variation_workflow, contract_performance_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-con"
__capability_id__ = "government_con"

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
