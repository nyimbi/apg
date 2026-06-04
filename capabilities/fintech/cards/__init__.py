"""APG Digital Cards capability.

Standalone package: ``pip install apg-fintech-cards``

Quick start::

    from apg_fintech_cards import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_cards
Provides      : card_program_governance, cardholder_card_lifecycle, tokenized_card_credentialing, card_authorization_control, card_dispute_workflow, card_agent_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-cards"
__capability_id__ = "fintech_cards"

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
