"""APG Ontology Management capability.

Standalone package: ``pip install apg-common-onto``

Quick start::

    from apg_common_onto import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : onto
Provides      : ontology_management, semantic_vocabulary_governance, ontology_agent_composition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-onto"
__capability_id__ = "onto"

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
