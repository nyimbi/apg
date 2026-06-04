"""APG Service Provisioning capability.

Standalone package: ``pip install apg-telecom-pro``

Quick start::

    from apg_telecom_pro import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_pro
Provides      : service_activation_workflow, network_resource_allocation, configuration_push_workflow, activation_confirmation_workflow, rollback_workflow, bulk_provisioning_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-pro"
__capability_id__ = "telecom_pro"

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
