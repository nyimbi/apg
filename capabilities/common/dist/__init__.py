"""APG Distributed Computing capability.

Standalone package: ``pip install apg-common-dist``

Quick start::

    from apg_common_dist import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : dist
Provides      : distributed_jobs, worker_pools, partitioned_execution, coordination, distributed_scaling, compute_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-dist"
__capability_id__ = "dist"

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
