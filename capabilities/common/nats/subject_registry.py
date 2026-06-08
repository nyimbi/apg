"""NATS subject naming conventions for APG events."""
from __future__ import annotations

APG_STREAM_NAME = "APG_EVENTS"
APG_SUBJECT_PREFIX = "apg.events"


def subject_for(capability_id: str, event_type: str) -> str:
	"""Return the canonical NATS subject for an APG event.

	Format: apg.events.{capability_id}.{event_type}
	Example: apg.events.ckm_wfa.workflow_started
	"""
	cap = capability_id.replace("-", "_").lower()
	evt = event_type.replace("-", "_").lower()
	return f"{APG_SUBJECT_PREFIX}.{cap}.{evt}"


def parse_subject(subject: str) -> tuple[str, str] | None:
	"""Parse a NATS subject into (capability_id, event_type).

	Returns None if subject does not match the APG convention.
	"""
	parts = subject.split(".")
	if len(parts) < 4 or parts[:2] != ["apg", "events"]:
		return None
	capability_id = parts[2]
	event_type = ".".join(parts[3:])
	return capability_id, event_type


# Wildcard subscriptions for common patterns
def subscribe_all_capability_events(capability_id: str) -> str:
	"""Wildcard subject for all events from one capability."""
	return f"{APG_SUBJECT_PREFIX}.{capability_id}.>"


def subscribe_event_type_all_capabilities(event_type: str) -> str:
	"""Wildcard subject for one event type across all capabilities."""
	return f"{APG_SUBJECT_PREFIX}.*.{event_type}"


WELL_KNOWN_EVENTS = {
	"workflow_started", "workflow_completed", "workflow_failed", "workflow_cancelled",
	"task_assigned", "task_completed", "task_escalated",
	"payment_received", "payment_failed", "payment_reversed",
	"phi_accessed", "phi_modified",
	"access_decision", "auth_failed",
	"notification_requested", "notification_delivered",
	"audit_event",
}
