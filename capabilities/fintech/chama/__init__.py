"""APG Chama & ROSCA Engine capability.

Standalone package: ``pip install apg-fintech-chama``

Quick start::

    from apg_fintech_chama import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_sacco")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_chama
Provides      : chama_management, rosca_rotation, group_lending, treasury_management, mobile_disbursement

Chamas are the dominant savings vehicle for 60%+ of East Africa's population.
No SAP/Oracle product addresses this domain — this is APG's unique Africa differentiator.
"""
from __future__ import annotations

__version__ = "1.0.0"
__package_name__ = "apg-fintech-chama"
__capability_id__ = "fintech_chama"

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
