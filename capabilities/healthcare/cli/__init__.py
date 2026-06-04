"""APG Clinical Management capability.

Standalone package: ``pip install apg-healthcare-cli``

Quick start::

    from apg_healthcare_cli import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_cli
Provides      : care_plan_management, clinical_workflow_orchestration, protocol_adherence_tracking, clinical_decision_support, care_team_management, clinical_handoff_management
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-cli"
__capability_id__ = "healthcare_cli"

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
