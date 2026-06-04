"""APG Pharmacovigilance capability.

Standalone package: ``pip install apg-pharma-pvi``

Quick start::

    from apg_pharma_pvi import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_pvi
Provides      : adverse_event_collection_workflow, case_processing_workflow, signal_detection_workflow, psur_generation_workflow, regulatory_reporting_workflow, literature_screening_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-pvi"
__capability_id__ = "pharma_pvi"

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
