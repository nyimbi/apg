"""APG Recommender Systems capability.

Standalone package: ``pip install apg-common-recs``

Quick start::

    from apg_common_recs import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : recs
Provides      : personalized_recommendations, ranking_policies, catalog_matching, interaction_datasets, model_training, model_deployments
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-recs"
__capability_id__ = "recs"

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
