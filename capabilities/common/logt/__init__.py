"""APG Logging and Tracing capability.

Standalone package: ``pip install apg-common-logt``

Quick start::

    from apg_common_logt import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : logt
Provides      : structured_logging, distributed_tracing, trace_correlation, log_search, diagnostic_retention, diagnostic_exports
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-logt"
__capability_id__ = "logt"

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
