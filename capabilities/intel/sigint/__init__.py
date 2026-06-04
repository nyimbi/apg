"""APG Signals Intelligence capability.

Standalone package: ``pip install apg-intel-sigint``

Quick start::

    from apg_intel_sigint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_sigint
Provides      : sigint_authority_workflow, sigint_source_workflow, sigint_collection_workflow, sigint_observation_workflow, sigint_processing_workflow, sigint_pattern_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-sigint"
__capability_id__ = "intel_sigint"

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
