"""APG FinTech Compliance Automation capability.

Standalone package: ``pip install apg-fintech-compliance``

Quick start::

    from apg_fintech_compliance import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_compliance
Provides      : compliance_obligation_workflow, compliance_control_workflow, compliance_check_workflow, compliance_evidence_workflow, compliance_attestation_workflow, compliance_issue_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-compliance"
__capability_id__ = "fintech_compliance"

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
