"""APG Radio Intelligence Listener capability.

Standalone package: ``pip install apg-intel-radio``

Quick start::

    from apg_intel_radio import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_radio
Provides      : radio_authority_workflow, radio_band_plan_workflow, radio_receiver_workflow, radio_collection_session_workflow, radio_observation_workflow, radio_classification_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-radio"
__capability_id__ = "intel_radio"

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
