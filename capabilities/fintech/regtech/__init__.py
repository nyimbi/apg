"""APG Regulatory Technology capability.

Standalone package: ``pip install apg-fintech-regtech``

Quick start::

    from apg_fintech_regtech import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_regtech
Provides      : regulatory_source_workflow, regulatory_change_workflow, regulatory_obligation_mapping_workflow, regulatory_policy_mapping_workflow, regulatory_impact_workflow, regulatory_filing_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-regtech"
__capability_id__ = "fintech_regtech"

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
