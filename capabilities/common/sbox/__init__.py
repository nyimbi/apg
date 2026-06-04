"""APG Sandbox/Testing Environment capability.

Standalone package: ``pip install apg-common-sbox``

Quick start::

    from apg_common_sbox import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : sbox
Provides      : sandbox_registry, isolation_profiles, test_runs, synthetic_datasets, safety_policy, sbox_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-sbox"
__capability_id__ = "sbox"

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
