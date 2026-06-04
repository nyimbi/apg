"""APG Geospatial Intelligence capability.

Standalone package: ``pip install apg-intel-geoint``

Quick start::

    from apg_intel_geoint import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_geoint
Provides      : geoint_authority_workflow, geoint_area_workflow, geoint_source_workflow, geoint_collection_workflow, geoint_observation_workflow, geoint_feature_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-geoint"
__capability_id__ = "intel_geoint"

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
