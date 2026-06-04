"""APG Self-Service BI capability.

Standalone package: ``pip install apg-bia-sbi``

Quick start::

    from apg_bia_sbi import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_sbi
Provides      : drag_drop_visual_builder, natural_language_queries, governed_data_catalogue, user_sandboxes, template_gallery, self_service_chart_creation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-sbi"
__capability_id__ = "bia_sbi"

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
