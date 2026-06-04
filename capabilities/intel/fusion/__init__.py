"""APG Intelligence Fusion capability.

Standalone package: ``pip install apg-intel-fusion``

Quick start::

    from apg_intel_fusion import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_fusion
Provides      : fusion_authority_workflow, fusion_workspace_workflow, fusion_source_workflow, fusion_artifact_workflow, fusion_correlation_workflow, fusion_hypothesis_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-fusion"
__capability_id__ = "intel_fusion"

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
