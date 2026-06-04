"""APG Dashboard Management capability.

Standalone package: ``pip install apg-bia-dsh``

Quick start::

    from apg_bia_dsh import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_dsh
Provides      : dashboard_creation, widget_library, real_time_data_binding, responsive_layout_engine, scheduled_snapshots, cross_widget_filtering
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-dsh"
__capability_id__ = "bia_dsh"

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
