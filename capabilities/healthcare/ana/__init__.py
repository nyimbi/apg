"""APG Clinical Analytics capability.

Standalone package: ``pip install apg-healthcare-ana``

Quick start::

    from apg_healthcare_ana import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : healthcare_ana
Provides      : population_health_analytics, clinical_outcomes_measurement, readmission_prediction, quality_indicator_tracking, cohort_management, clinical_benchmarking
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-healthcare-ana"
__capability_id__ = "healthcare_ana"

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
