"""APG Enterprise Asset Management capability.

Standalone package: ``pip install apg-eam-ast``

Quick start::

    from apg_eam_ast import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : eam_ast
Provides      : asset_registry_lifecycle, asset_location_hierarchy, criticality_and_condition_management, maintenance_plan_lifecycle, work_order_lifecycle, inspection_and_condition_readings
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-eam-ast"
__capability_id__ = "eam_ast"

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
