"""Typed IntegrationEvent envelope for all cross-capability NATS messages.

Every service.py mutation method that publishes to NATS MUST use this envelope
so consumers can rely on a stable, typed contract rather than ad-hoc dicts.

Subject format: apg.events.{capability_id}.{event_type}
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
	from situ_cloudevents._uuid7 import uuid7str  # type: ignore[import]
except ImportError:
	from uuid6 import uuid7  # type: ignore[import]

	def uuid7str() -> str:
		return str(uuid7())


class IntegrationEvent(BaseModel):
	"""Canonical envelope for all APG cross-capability events.

	Published to: apg.events.{capability_id}.{event_type}
	Consumed by: any capability declaring a matching 'subscribes' entry.
	"""
	model_config = ConfigDict(extra="forbid", populate_by_name=True)

	capability_id: str = Field(description="Emitting capability, e.g. 'fintech_gateway'")
	event_type: str = Field(description="Event name, e.g. 'payment_authorized'")
	entity_type: str = Field(description="Domain entity type, e.g. 'payment_intent'")
	entity_id: str = Field(description="Local entity ID within the emitting capability")
	canonical_entity_id: str | None = Field(
		default=None,
		description="MDM canonical UUID — set after MDM resolution, None for new entities",
	)
	tenant_id: str = Field(description="Tenant scope for multi-tenancy isolation")
	actor_id: str = Field(default="system", description="User or service that triggered the event")
	payload: dict[str, Any] = Field(
		default_factory=dict,
		description="Event-specific data; schema declared in capability_contract.py publishes[]",
	)
	correlation_id: str = Field(
		default_factory=uuid7str,
		description="Trace ID propagated across all events in the same request chain",
	)
	causation_id: str | None = Field(
		default=None,
		description="correlation_id of the upstream event that caused this one",
	)
	occurred_at: datetime = Field(
		default_factory=lambda: datetime.now(timezone.utc),
		description="UTC timestamp when the domain event occurred",
	)
	schema_version: str = Field(
		default="1.0",
		description="Envelope schema version for forward-compatibility",
	)

	def subject(self) -> str:
		"""Return the canonical NATS subject for this event."""
		from .subject_registry import subject_for
		return subject_for(self.capability_id, self.event_type)

	def msg_id(self) -> str:
		"""Deterministic deduplication key for NATS Msg-Id header."""
		return f"{self.tenant_id}-{self.entity_id}-{self.event_type}-{self.occurred_at.isoformat()}"

	@classmethod
	def from_audit_call(
		cls,
		*,
		capability_id: str,
		event_type: str,
		entity_type: str,
		actor_id: str,
		tenant_id: str,
		resource_id: str,
		details: dict[str, Any],
		correlation_id: str | None = None,
		causation_id: str | None = None,
	) -> "IntegrationEvent":
		"""Build an IntegrationEvent from the legacy log_event() parameter shape."""
		return cls(
			capability_id=capability_id,
			event_type=event_type,
			entity_type=entity_type,
			entity_id=resource_id,
			tenant_id=tenant_id,
			actor_id=actor_id,
			payload=details,
			correlation_id=correlation_id or uuid7str(),
			causation_id=causation_id,
		)
