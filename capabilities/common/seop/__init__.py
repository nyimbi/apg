"""APG Security Operations capability.

Standalone package: ``pip install apg-common-seop``

Quick start::

    from apg_common_seop import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : seop
Provides      : detection_pipeline, incident_response, threat_triage, response_playbooks, security_posture, seop_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-seop"
__capability_id__ = "seop"

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
