"""APG Citizen Services Portal capability.

Standalone package: ``pip install apg-government-csr``

Quick start::

    from apg_government_csr import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : government_csr
Provides      : citizen_self_service_workflow, service_application_workflow, application_status_tracking_workflow, epayment_workflow, document_verification_workflow, service_notification_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-government-csr"
__capability_id__ = "government_csr"

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
