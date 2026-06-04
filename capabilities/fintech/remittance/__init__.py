"""APG Cross-Border Remittance capability.

Standalone package: ``pip install apg-fintech-remittance``

Quick start::

    from apg_fintech_remittance import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_remittance
Provides      : remittance_corridor_governance, remittance_quote_lifecycle, cross_border_transfer_workflow, remittance_payout_workflow, remittance_refund_workflow, remittance_agent_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-remittance"
__capability_id__ = "fintech_remittance"

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
