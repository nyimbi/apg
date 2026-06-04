"""APG Audit Management capability.

Standalone package: ``pip install apg-grc-aud``

Quick start::

    from apg_grc_aud import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : grc_aud
Provides      : audit_program_lifecycle, audit_finding_lifecycle, audit_evidence_workflow, audit_report_workflow, audit_dashboard_service, audit_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-grc-aud"
__capability_id__ = "grc_aud"

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
