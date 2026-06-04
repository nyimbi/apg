"""APG Exploration Data Management capability.

Standalone package: ``pip install apg-mining-exp``

Quick start::

    from apg_mining_exp import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mining_exp
Provides      : drillhole_collar_management, downhole_survey_management, lithology_logging, assay_data_management, qaqc_monitoring, resource_estimation_workflow
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-mining-exp"
__capability_id__ = "mining_exp"

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
