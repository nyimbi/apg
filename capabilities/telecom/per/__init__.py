"""APG Performance Management capability.

Standalone package: ``pip install apg-telecom-per``

Quick start::

    from apg_telecom_per import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : telecom_per
Provides      : kpi_monitoring_workflow, sla_compliance_workflow, capacity_utilisation_workflow, trend_reporting_workflow, performance_reporting_workflow, threshold_management_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-telecom-per"
__capability_id__ = "telecom_per"

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
