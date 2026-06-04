"""APG Mine Safety & Compliance capability.

Standalone package: ``pip install apg-mining-saf``

Quick start::

    from apg_mining_saf import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mining_saf
Provides      : incident_reporting_workflow, hazard_identification_workflow, risk_register_management, permit_to_work_workflow, corrective_action_tracking, compliance_register_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mining-saf"
__capability_id__ = "mining_saf"

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
