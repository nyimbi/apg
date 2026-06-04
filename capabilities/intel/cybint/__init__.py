"""APG Cyber Intelligence capability.

Standalone package: ``pip install apg-intel-cybint``

Quick start::

    from apg_intel_cybint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_cybint
Provides      : cybint_authority_workflow, cybint_indicator_workflow, cybint_sighting_workflow, cybint_enrichment_workflow, cybint_threat_profile_workflow, cybint_risk_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-cybint"
__capability_id__ = "intel_cybint"

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
