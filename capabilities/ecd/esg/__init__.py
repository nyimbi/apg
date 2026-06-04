"""APG Sustainability and ESG Management capability.

Standalone package: ``pip install apg-ecd-esg``

Quick start::

    from apg_ecd_esg import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ecd_esg
Provides      : esg_profile_lifecycle, esg_framework_lifecycle, esg_metric_lifecycle, esg_measurement_lifecycle, esg_target_lifecycle, esg_supplier_assessment_lifecycle
"""
from __future__ import annotations

__version__  = "2.1.0"
__package_name__ = "apg-ecd-esg"
__capability_id__ = "ecd_esg"

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
