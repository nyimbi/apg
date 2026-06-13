"""APG SaaS Billing Engine capability.

Standalone package: ``pip install apg-common-sbl``

Quick start::

    from apg_common_sbl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : common_sbl
Provides      : subscription_management, usage_metering, invoice_generation, tenant_provisioning, billing_analytics
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-sbl"
__capability_id__ = "common_sbl"

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
