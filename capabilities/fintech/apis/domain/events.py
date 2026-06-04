"""Domain events for Banking APIs.

Events are emitted to the capability event stream via Bytewax whenever state
changes occur. Subscribe to these events for integration, auditing, and
downstream capability composition.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all Banking APIs domain events."""
    event_type: str
    tenant_id: str
    actor_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "capability_id": "fintech_apis",
        }
