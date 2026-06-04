"""APG Telemedicine capability.

Standalone package: ``pip install apg-healthcare-tel``

Quick start::

    from apg_healthcare_tel import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_tel
Provides      : virtual_consultation_booking, video_session_management, remote_patient_monitoring, prescription_transmission, telehealth_billing, patient_consent_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-tel"
__capability_id__ = "healthcare_tel"

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
