"""APG Remote Workforce capability.

Standalone package: ``pip install apg-mob-rwf``

Quick start::

    from apg_mob_rwf import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mob_rwf
Provides      : remote_work_policy_management, vpn_access_governance, productivity_tracking_workflow, equipment_requisition_workflow, digital_onboarding_workflow, remote_compliance_monitoring
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mob-rwf"
__capability_id__ = "mob_rwf"

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
