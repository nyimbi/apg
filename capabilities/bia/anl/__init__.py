"""APG Analytics Engine capability.

Standalone package: ``pip install apg-bia-anl``

Quick start::

    from apg_bia_anl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_anl
Provides      : ad_hoc_query_execution, olap_cube_management, metric_definition_registry, analytical_data_access, query_result_cache, datasource_connectivity
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-anl"
__capability_id__ = "bia_anl"

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
