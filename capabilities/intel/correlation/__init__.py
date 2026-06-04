"""APG Data Correlation capability.

Standalone package: ``pip install apg-intel-correlation``

Quick start::

    from apg_intel_correlation import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_correlation
Provides      : correlation_authority_workflow, correlation_workspace_workflow, correlation_source_workflow, correlation_entity_workflow, correlation_observation_workflow, correlation_rule_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-correlation"
__capability_id__ = "intel_correlation"

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
