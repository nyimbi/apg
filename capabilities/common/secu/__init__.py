"""APG Security Framework capability.

Standalone package: ``pip install apg-common-secu``

Quick start::

    from apg_common_secu import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : secu
Provides      : risk_assessment, threat_detection, security_policies, compliance_automation, incident_response_governance, security_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-secu"
__capability_id__ = "secu"

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

# Backward-compatibility stub

class SecurityLevel:
    PUBLIC = "public"; INTERNAL = "internal"; CONFIDENTIAL = "confidential"; RESTRICTED = "restricted"; SECRET = "secret"

class RiskLevel:
    LOW = "low"; MEDIUM = "medium"; HIGH = "high"; CRITICAL = "critical"
