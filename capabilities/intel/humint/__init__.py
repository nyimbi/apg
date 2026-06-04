"""APG Human Intelligence capability.

Standalone package: ``pip install apg-intel-humint``

Quick start::

    from apg_intel_humint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_humint
Provides      : humint_authority_workflow, humint_source_workflow, humint_contact_plan_workflow, humint_contact_report_workflow, humint_debriefing_workflow, humint_reliability_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-humint"
__capability_id__ = "intel_humint"

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
