"""APG Pharmaceutical Manufacturing capability.

Standalone package: ``pip install apg-pharma-mfg``

Quick start::

    from apg_pharma_mfg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_mfg
Provides      : batch_record_management_workflow, manufacturing_execution_workflow, equipment_qualification_workflow, yield_management_workflow, deviation_management_workflow, gmp_compliance_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-mfg"
__capability_id__ = "pharma_mfg"

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
