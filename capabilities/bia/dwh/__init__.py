"""APG Data Warehouse capability.

Standalone package: ``pip install apg-bia-dwh``

Quick start::

    from apg_bia_dwh import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : bia_dwh
Provides      : dimensional_schema_management, star_snowflake_schema_design, etl_orchestration, data_partitioning, data_quality_enforcement, lineage_tracking
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-bia-dwh"
__capability_id__ = "bia_dwh"

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
