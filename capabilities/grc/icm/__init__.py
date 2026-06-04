"""APG Incident and Case Management capability.

Standalone package: ``pip install apg-grc-icm``

Quick start::

    from apg_grc_icm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : grc_icm
Provides      : incident_lifecycle_management, case_management_workflow, incident_evidence_workflow, regulatory_notification_workflow, post_incident_review_workflow, icm_dashboard_service
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-grc-icm"
__capability_id__ = "grc_icm"

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
