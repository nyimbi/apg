"""APG Report Builder capability.

Standalone package: ``pip install apg-bia-rpt``

Quick start::

    from apg_bia_rpt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_rpt
Provides      : parameterised_report_authoring, report_scheduling, report_distribution, multi_format_export, report_audit_trail, report_template_library
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-rpt"
__capability_id__ = "bia_rpt"

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
