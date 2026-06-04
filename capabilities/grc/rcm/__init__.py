"""APG grc_rcm capability.

Standalone package: ``pip install apg-grc-rcm``

Quick start::

    from apg_grc_rcm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : grc_rcm
Provides      : risk_register_lifecycle, control_library_lifecycle, compliance_obligation_lifecycle, control_assessment_workflow, evidence_management_workflow, issue_remediation_workflow
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-grc-rcm"
__capability_id__ = "grc_rcm"

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
