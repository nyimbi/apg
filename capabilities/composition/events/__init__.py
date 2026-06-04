"""APG Event Streaming Bus capability.

Standalone package: ``pip install apg-composition-events``

Quick start::

    from apg_composition_events import get_capability_contract, evaluate_capability_rules

    contract = get_capability_contract(tenant_id="my_org")
    result   = evaluate_capability_rules({"tenant_context_present": True, "operation_type": "read"})

Capability ID : composition_events
Provides      : event_stream_registry, bytewax_event_publishing, event_schema_registry, subscription_lifecycle, stream_processor_topology, dead_letter_operations
"""
from __future__ import annotations

__version__  = "1.0.0"
__package_name__ = "apg-composition-events"
__capability_id__ = "composition_events"

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
