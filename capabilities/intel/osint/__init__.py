"""APG Open Source Intelligence capability.

Standalone package: ``pip install apg-intel-osint``

Quick start::

    from apg_intel_osint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_osint
Provides      : osint_requirement_workflow, osint_source_workflow, osint_collection_plan_workflow, osint_evidence_workflow, osint_triage_workflow, osint_assessment_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-osint"
__capability_id__ = "intel_osint"

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
