"""APG Knowledge Graph capability.

Standalone package: ``pip install apg-common-kngr``

Quick start::

    from apg_common_kngr import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : kngr
Provides      : knowledge_graph, semantic_context, knowledge_agent_composition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-kngr"
__capability_id__ = "kngr"

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
