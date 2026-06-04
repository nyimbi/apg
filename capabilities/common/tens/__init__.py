"""APG Tenants Legacy capability.

Standalone package: ``pip install apg-common-tens``

Quick start::

    from apg_common_tens import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : tens
Provides      : legacy_tenant_registry, tenant_mapping, migration_controls, access_boundaries, deprecation_governance, tens_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-tens"
__capability_id__ = "tens"

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
