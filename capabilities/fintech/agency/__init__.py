"""APG Agency Banking capability.

Standalone package: ``pip install apg-fintech-agency``

Quick start::

    from apg_fintech_agency import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_agency
Provides      : agency_program_governance, agency_outlet_lifecycle, agency_agent_accreditation, agency_float_management, agency_customer_workflow, agency_transaction_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-agency"
__capability_id__ = "fintech_agency"

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
