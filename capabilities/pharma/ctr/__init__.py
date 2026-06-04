"""APG Clinical Trials Management capability.

Standalone package: ``pip install apg-pharma-ctr``

Quick start::

    from apg_pharma_ctr import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_ctr
Provides      : trial_protocol_workflow, site_selection_workflow, patient_randomisation_workflow, adverse_event_workflow, clinical_data_management_workflow, regulatory_submission_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-ctr"
__capability_id__ = "pharma_ctr"

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
