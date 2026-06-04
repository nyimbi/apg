"""APG Custom Scripting Engine capability.

Standalone package: ``pip install apg-common-scpt``

Quick start::

    from apg_common_scpt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : scpt
Provides      : script_registry, secure_sandbox, workflow_extensions, package_policy, script_execution, scripting_agent_composition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-scpt"
__capability_id__ = "scpt"

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
