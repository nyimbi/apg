"""APG Network Management capability.

Standalone package: ``pip install apg-telecom-net``

Quick start::

    from apg_telecom_net import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_net
Provides      : fault_management_workflow, performance_management_workflow, configuration_management_workflow, sla_monitoring_workflow, noc_operations_workflow, alarm_correlation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-net"
__capability_id__ = "telecom_net"

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
