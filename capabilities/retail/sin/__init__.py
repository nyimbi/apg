"""APG Store Intelligence capability.

Standalone package: ``pip install apg-retail-sin``

Quick start::

    from apg_retail_sin import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : retail_sin
Provides      : store_foot_traffic_analytics, planogram_compliance_monitoring, shelf_availability_alerting, store_conversion_optimisation, store_performance_benchmarking, zone_heatmap_analytics
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-retail-sin"
__capability_id__ = "retail_sin"

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
