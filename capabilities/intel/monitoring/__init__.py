"""APG Real-Time Monitoring capability.

Standalone package: ``pip install apg-intel-monitoring``

Quick start::

    from apg_intel_monitoring import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : intel_monitoring
Provides      : monitoring_authority_workflow, monitoring_policy_workflow, monitoring_source_workflow, monitoring_watch_workflow, monitoring_event_workflow, monitoring_signal_workflow
"""
from __future__ import annotations

__version__  = "1.1.0"
__package_name__ = "apg-intel-monitoring"
__capability_id__ = "intel_monitoring"

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
