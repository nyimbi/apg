"""APG Project Planning & Scheduling capability.

Standalone package: ``pip install apg-ppm-pps``

Quick start::

    from apg_ppm_pps import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : ppm_pps
Provides      : wbs_creation_and_management, critical_path_analysis, resource_levelling, dependency_management, timeline_management, schedule_optimisation
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-ppm-pps"
__capability_id__ = "ppm_pps"

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
