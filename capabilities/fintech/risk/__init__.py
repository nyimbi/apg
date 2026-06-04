"""APG FinTech Risk Management capability.

Standalone package: ``pip install apg-fintech-risk``

Quick start::

    from apg_fintech_risk import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_risk
Provides      : risk_appetite_workflow, risk_profile_workflow, risk_exposure_workflow, risk_control_workflow, risk_stress_testing_workflow, risk_limit_breach_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-risk"
__capability_id__ = "fintech_risk"

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
