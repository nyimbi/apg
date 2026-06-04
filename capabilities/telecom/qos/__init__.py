"""APG Quality of Service capability.

Standalone package: ``pip install apg-telecom-qos``

Quick start::

    from apg_telecom_qos import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_qos
Provides      : qos_policy_management_workflow, traffic_prioritisation_workflow, sla_enforcement_workflow, degradation_detection_workflow, root_cause_analysis_workflow, auto_remediation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-qos"
__capability_id__ = "telecom_qos"

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
