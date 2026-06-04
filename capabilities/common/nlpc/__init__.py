"""APG NLP Core capability.

Standalone package: ``pip install apg-common-nlpc``

Quick start::

    from apg_common_nlpc import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : nlpc
Provides      : text_intelligence, multilingual_processing, nlp_agent_composition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-nlpc"
__capability_id__ = "nlpc"

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
