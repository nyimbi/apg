"""APG Accessibility Services capability.

Standalone package: ``pip install apg-common-accs``

Quick start::

    from apg_common_accs import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : accs
Provides      : accessibility_audits, remediation_workflows, accessibility_exceptions, assistive_metadata, media_accessibility, standards_governance
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-accs"
__capability_id__ = "accs"

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
