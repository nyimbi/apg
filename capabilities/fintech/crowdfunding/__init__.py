"""APG Crowdfunding Platform capability.

Standalone package: ``pip install apg-fintech-crowdfunding``

Quick start::

    from apg_fintech_crowdfunding import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : fintech_crowdfunding
Provides      : crowdfunding_issuer_workflow, crowdfunding_campaign_workflow, crowdfunding_disclosure_workflow, crowdfunding_commitment_workflow, crowdfunding_escrow_workflow, crowdfunding_milestone_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-fintech-crowdfunding"
__capability_id__ = "fintech_crowdfunding"

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
