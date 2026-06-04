"""APG Environment Management capability.

Standalone package: ``pip install apg-common-envm``

Quick start::

    from apg_common_envm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : envm
Provides      : environment_inventory, environment_promotion, configuration_drift, secret_scopes, environment_policy, envm_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-envm"
__capability_id__ = "envm"

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
