"""APG Product Registration capability.

Standalone package: ``pip install apg-pharma-reg``

Quick start::

    from apg_pharma_reg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_reg
Provides      : registration_application_workflow, dossier_compilation_workflow, authority_interaction_workflow, approval_tracking_workflow, lifecycle_maintenance_workflow, variation_management_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-reg"
__capability_id__ = "pharma_reg"

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
