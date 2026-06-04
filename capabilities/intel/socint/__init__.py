"""APG Social Media Intelligence capability.

Standalone package: ``pip install apg-intel-socint``

Quick start::

    from apg_intel_socint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_socint
Provides      : socint_authority_workflow, socint_topic_workflow, socint_source_workflow, socint_post_workflow, socint_signal_workflow, socint_influence_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-socint"
__capability_id__ = "intel_socint"

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
