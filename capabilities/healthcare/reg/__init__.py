"""APG Healthcare Regulatory capability.

Standalone package: ``pip install apg-healthcare-reg``

Quick start::

    from apg_healthcare_reg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_reg
Provides      : facility_licensing_management, accreditation_management, incident_reporting, hipaa_compliance_tracking, regulatory_submission_management, audit_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-reg"
__capability_id__ = "healthcare_reg"

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
