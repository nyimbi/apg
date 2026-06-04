"""APG Regulatory Compliance capability.

Standalone package: ``pip install apg-pharma-rec``

Quick start::

    from apg_pharma_rec import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : pharma_rec
Provides      : regulatory_compliance_monitoring_workflow, inspection_readiness_workflow, label_management_workflow, post_market_surveillance_workflow, regulatory_intelligence_workflow, commitment_tracking_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-pharma-rec"
__capability_id__ = "pharma_rec"

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
