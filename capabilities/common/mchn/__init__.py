"""APG Multi-Channel Output capability.

Standalone package: ``pip install apg-common-mchn``

Quick start::

    from apg_common_mchn import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mchn
Provides      : channel_routing, format_rendering, output_templates, delivery_policy, delivery_receipts, omnichannel_analytics
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-mchn"
__capability_id__ = "mchn"

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
