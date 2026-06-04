"""APG Commercial Operations capability.

Standalone package: ``pip install apg-pharma-com``

Quick start::

    from apg_pharma_com import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_com
Provides      : territory_management_workflow, sales_rep_management_workflow, call_activity_workflow, sample_management_workflow, hcp_interaction_workflow, commercial_plan_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-com"
__capability_id__ = "pharma_com"

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
