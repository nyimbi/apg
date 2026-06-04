"""APG Payment Switch capability.

Standalone package: ``pip install apg-fintech-switch``

Quick start::

    from apg_fintech_switch import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_switch
Provides      : iso8583_message_switching, payment_routing_engine, channel_key_management, pin_block_translation, mac_generation_verification, mobile_money_switching
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-switch"
__capability_id__ = "fintech_switch"

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
