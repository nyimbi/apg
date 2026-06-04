"""Domain events for Budgeting and Forecasting.

Events are emitted to the capability event stream whenever state
changes occur. Subscribe to these events for integration, auditing,
and downstream capability composition.

© 2025 Datacraft. Author: Nyimbi Odero <nyimbi@gmail.com>
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DomainEvent:
	"""Base class for all BFC domain events."""

	event_type: str
	tenant_id: str
	actor_id: str
	timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"event_type": self.event_type,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"timestamp": self.timestamp.isoformat(),
			"payload": self.payload,
			"capability_id": "bfc_budgeting_forecasting",
		}
