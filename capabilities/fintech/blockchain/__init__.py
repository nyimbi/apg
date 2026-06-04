"""APG Blockchain Services capability.

Standalone package: ``pip install apg-fintech-blockchain``

Quick start::

    from apg_fintech_blockchain import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_blockchain
Provides      : blockchain_network_workflow, blockchain_wallet_workflow, smart_contract_workflow, chain_transaction_workflow, evidence_anchor_workflow, oracle_feed_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-blockchain"
__capability_id__ = "fintech_blockchain"

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
