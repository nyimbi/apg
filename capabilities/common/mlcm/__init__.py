"""APG AI Model Lifecycle Management capability.

Standalone package: ``pip install apg-common-mlcm``

Quick start::

    from apg_common_mlcm import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : mlcm
Provides      : model_lifecycle, model_governance, model_lifecycle_agent_composition
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-mlcm"
__capability_id__ = "mlcm"

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
