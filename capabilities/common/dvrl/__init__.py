"""APG Data Virtualization capability.

Standalone package: ``pip install apg-common-dvrl``

Quick start::

    from apg_common_dvrl import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : dvrl
Provides      : data_virtualization, federated_query_lifecycle, virtualization_agent_composition, review_evidence
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-common-dvrl"
__capability_id__ = "dvrl"

from .capability_contract import (  # noqa: E402
    get_capability_contract,
    evaluate_capability_rules,
)

from datetime import datetime, timezone as _tz


async def _log_info(message: str, context: dict = None) -> None:
    """Log informational message for DVRL operations."""
    ts = datetime.now(_tz.utc).isoformat()
    print(f"[{ts}] DVRL INFO: {message}")


async def _log_error(message: str, error: Exception = None) -> None:
    """Log error message for DVRL operations."""
    ts = datetime.now(_tz.utc).isoformat()
    suffix = f" | Error: {error}" if error else ""
    print(f"[{ts}] DVRL ERROR: {message}{suffix}")


async def _log_warning(message: str, context: dict = None) -> None:
    """Log warning message for DVRL operations."""
    ts = datetime.now(_tz.utc).isoformat()
    print(f"[{ts}] DVRL WARN: {message}")


__all__ = [
    "__version__",
    "__capability_id__",
    "get_capability_contract",
    "evaluate_capability_rules",
    "_log_info",
    "_log_error",
    "_log_warning",
]
