"""APG Blockchain Ledger Services capability.

Standalone package: ``pip install apg-common-bclg``

Quick start::

    from apg_common_bclg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bclg
Provides      : ledger_registry, transaction_governance, smart_contract_governance, key_custody_governance, ledger_audit, ledger_agents
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-bclg"
__capability_id__ = "bclg"

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
