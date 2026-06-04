"""APG Intelligence Reporting capability.

Standalone package: ``pip install apg-intel-reporting``

Quick start::

    from apg_intel_reporting import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_reporting
Provides      : reporting_authority_workflow, reporting_workspace_workflow, reporting_template_workflow, reporting_product_workflow, reporting_section_workflow, reporting_citation_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-reporting"
__capability_id__ = "intel_reporting"

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
