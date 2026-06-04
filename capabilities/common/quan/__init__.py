"""APG Quantum Computing capability.

Standalone package: ``pip install apg-common-quan``

Quick start::

    from apg_common_quan import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : quan
Provides      : quantum_backend_registry, circuit_management, quantum_job_orchestration, result_analysis, post_quantum_governance, quan_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-quan"
__capability_id__ = "quan"

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
