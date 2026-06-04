"""APG Dark Web Monitoring capability.

Standalone package: ``pip install apg-intel-darkweb``

Quick start::

    from apg_intel_darkweb import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_darkweb
Provides      : darkweb_authority_workflow, darkweb_program_workflow, darkweb_source_workflow, darkweb_observation_workflow, darkweb_indicator_workflow, darkweb_marketplace_risk_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-darkweb"
__capability_id__ = "intel_darkweb"

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
