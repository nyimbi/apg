"""APG Advanced CRM Analytics capability.

Standalone package: ``pip install apg-crm-adv``

Quick start::

    from apg_crm_adv import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : crm_adv
Provides      : account_lifecycle, contact_relationship_management, lead_scoring_and_assignment, sales_pipeline_management, activity_timeline, campaign_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-crm-adv"
__capability_id__ = "crm_adv"

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
