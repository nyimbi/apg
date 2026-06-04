"""Domain events for Workflow Orchestration.

Events are emitted to the capability event stream via Bytewax whenever state
changes occur.  Subscribe to these events for integration, auditing, and
downstream capability composition.

All events are frozen dataclasses — immutable value objects safe to pass
across thread/process boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
	return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DomainEvent:
	"""Base class for all Workflow Orchestration domain events."""
	event_type: str
	tenant_id: str
	actor_id: str
	timestamp: datetime = field(default_factory=_utc_now)
	payload: dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> dict[str, Any]:
		return {
			"event_type": self.event_type,
			"tenant_id": self.tenant_id,
			"actor_id": self.actor_id,
			"timestamp": self.timestamp.isoformat(),
			"payload": self.payload,
			"capability_id": "wflo",
		}


# ─────────────────────────────────────────────────────────────────────────────
# Definition lifecycle events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DefinitionCreatedEvent(DomainEvent):
	"""Emitted when a new workflow definition is created."""
	event_type: str = "wflo.definition.created"


@dataclass(frozen=True)
class DefinitionPublishedEvent(DomainEvent):
	"""Emitted when a workflow definition is published."""
	event_type: str = "wflo.definition.published"


@dataclass(frozen=True)
class DefinitionDeprecatedEvent(DomainEvent):
	"""Emitted when a workflow definition is deprecated."""
	event_type: str = "wflo.definition.deprecated"


@dataclass(frozen=True)
class DefinitionRetiredEvent(DomainEvent):
	"""Emitted when a workflow definition is retired."""
	event_type: str = "wflo.definition.retired"


# ─────────────────────────────────────────────────────────────────────────────
# Instance lifecycle events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InstanceStartedEvent(DomainEvent):
	"""Emitted when a workflow instance begins execution."""
	event_type: str = "wflo.instance.started"


@dataclass(frozen=True)
class InstanceSuspendedEvent(DomainEvent):
	"""Emitted when a workflow instance is suspended."""
	event_type: str = "wflo.instance.suspended"


@dataclass(frozen=True)
class InstanceResumedEvent(DomainEvent):
	"""Emitted when a suspended workflow instance is resumed."""
	event_type: str = "wflo.instance.resumed"


@dataclass(frozen=True)
class InstanceCompletedEvent(DomainEvent):
	"""Emitted when a workflow instance completes successfully."""
	event_type: str = "wflo.instance.completed"


@dataclass(frozen=True)
class InstanceFailedEvent(DomainEvent):
	"""Emitted when a workflow instance fails."""
	event_type: str = "wflo.instance.failed"


@dataclass(frozen=True)
class InstanceCancelledEvent(DomainEvent):
	"""Emitted when a workflow instance is cancelled."""
	event_type: str = "wflo.instance.cancelled"


@dataclass(frozen=True)
class InstanceMigratedEvent(DomainEvent):
	"""Emitted when a running instance is migrated to a new definition version."""
	event_type: str = "wflo.instance.migrated"


@dataclass(frozen=True)
class SLABreachedEvent(DomainEvent):
	"""Emitted when an instance breaches its SLA deadline."""
	event_type: str = "wflo.instance.sla_breached"


# ─────────────────────────────────────────────────────────────────────────────
# Task events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TaskCreatedEvent(DomainEvent):
	"""Emitted when a task is created in a workflow instance."""
	event_type: str = "wflo.task.created"


@dataclass(frozen=True)
class TaskClaimedEvent(DomainEvent):
	"""Emitted when a user claims a task."""
	event_type: str = "wflo.task.claimed"


@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
	"""Emitted when a task is completed."""
	event_type: str = "wflo.task.completed"


@dataclass(frozen=True)
class TaskEscalatedEvent(DomainEvent):
	"""Emitted when a task is escalated."""
	event_type: str = "wflo.task.escalated"


@dataclass(frozen=True)
class TaskTimedOutEvent(DomainEvent):
	"""Emitted when a task exceeds its SLA without completion."""
	event_type: str = "wflo.task.timed_out"


# ─────────────────────────────────────────────────────────────────────────────
# Timer events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimerFiredEvent(DomainEvent):
	"""Emitted when a workflow timer fires."""
	event_type: str = "wflo.timer.fired"


@dataclass(frozen=True)
class TimerCancelledEvent(DomainEvent):
	"""Emitted when a workflow timer is cancelled."""
	event_type: str = "wflo.timer.cancelled"


# ─────────────────────────────────────────────────────────────────────────────
# Gateway events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GatewayEvaluatedEvent(DomainEvent):
	"""Emitted when a gateway condition is evaluated."""
	event_type: str = "wflo.gateway.evaluated"


# ─────────────────────────────────────────────────────────────────────────────
# Boundary / compensation / escalation events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryEventTriggeredEvent(DomainEvent):
	"""Emitted when a boundary event is triggered on a task."""
	event_type: str = "wflo.boundary_event.triggered"


@dataclass(frozen=True)
class EscalationCreatedEvent(DomainEvent):
	"""Emitted when a formal escalation record is created."""
	event_type: str = "wflo.escalation.created"


@dataclass(frozen=True)
class EscalationResolvedEvent(DomainEvent):
	"""Emitted when an escalation is resolved."""
	event_type: str = "wflo.escalation.resolved"


@dataclass(frozen=True)
class CompensationTriggeredEvent(DomainEvent):
	"""Emitted when compensation is triggered for a failed instance."""
	event_type: str = "wflo.compensation.triggered"


@dataclass(frozen=True)
class CompensationCompletedEvent(DomainEvent):
	"""Emitted when compensation completes."""
	event_type: str = "wflo.compensation.completed"


@dataclass(frozen=True)
class CompensationFailedEvent(DomainEvent):
	"""Emitted when compensation itself fails."""
	event_type: str = "wflo.compensation.failed"


# ─────────────────────────────────────────────────────────────────────────────
# Variable events
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VariableSetEvent(DomainEvent):
	"""Emitted when a workflow variable is created or updated."""
	event_type: str = "wflo.variable.set"


__all__ = [
	"DomainEvent",
	"DefinitionCreatedEvent",
	"DefinitionPublishedEvent",
	"DefinitionDeprecatedEvent",
	"DefinitionRetiredEvent",
	"InstanceStartedEvent",
	"InstanceSuspendedEvent",
	"InstanceResumedEvent",
	"InstanceCompletedEvent",
	"InstanceFailedEvent",
	"InstanceCancelledEvent",
	"InstanceMigratedEvent",
	"SLABreachedEvent",
	"TaskCreatedEvent",
	"TaskClaimedEvent",
	"TaskCompletedEvent",
	"TaskEscalatedEvent",
	"TaskTimedOutEvent",
	"TimerFiredEvent",
	"TimerCancelledEvent",
	"GatewayEvaluatedEvent",
	"BoundaryEventTriggeredEvent",
	"EscalationCreatedEvent",
	"EscalationResolvedEvent",
	"CompensationTriggeredEvent",
	"CompensationCompletedEvent",
	"CompensationFailedEvent",
	"VariableSetEvent",
]
