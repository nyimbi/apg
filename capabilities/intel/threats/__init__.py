"""APG Threat Intelligence capability.

Standalone package: ``pip install apg-intel-threats``

Quick start::

    from apg_intel_threats import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_threats
Provides      : threat_authority_workflow, threat_workspace_workflow, threat_source_workflow, threat_indicator_workflow, threat_actor_workflow, threat_campaign_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-threats"
__capability_id__ = "intel_threats"

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
