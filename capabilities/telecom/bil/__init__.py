"""APG Telecom Billing capability.

Standalone package: ``pip install apg-telecom-bil``

Quick start::

    from apg_telecom_bil import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_bil
Provides      : mediation_workflow, rating_workflow, charging_workflow, invoice_workflow, bill_cycle_management, dunning_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-bil"
__capability_id__ = "telecom_bil"

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
