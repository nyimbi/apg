"""APG Quality Management System capability.

Standalone package: ``pip install apg-pharma-qms``

Quick start::

    from apg_pharma_qms import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_qms
Provides      : change_control_workflow, capa_management_workflow, deviation_management_workflow, document_control_workflow, audit_management_workflow, validation_lifecycle_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-qms"
__capability_id__ = "pharma_qms"

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
