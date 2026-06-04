"""Domain events for Monitoring and Observability.

Events are emitted to the Bytewax capability event stream whenever state
changes occur. Subscribe to these events for integration, auditing, and
downstream capability composition.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ─── Base ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DomainEvent:
	"""Base class for all Monitoring and Observability domain events."""

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
			"capability_id": "moni",
		}


# ─── Metric events ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MetricRecorded(DomainEvent):
	"""Emitted when a metric data point is successfully ingested."""

	event_type: str = field(default="metric_recorded", init=False)

	@classmethod
	def from_metric(
		cls,
		tenant_id: str,
		actor_id: str,
		metric_name: str,
		value: float,
		source: str,
		labels: dict[str, str] | None = None,
	) -> "MetricRecorded":
		return cls(
			tenant_id=tenant_id,
			actor_id=actor_id,
			payload={
				"metric_name": metric_name,
				"value": value,
				"source": source,
				"labels": labels or {},
			},
		)


@dataclass(frozen=True)
class MetricThresholdBreached(DomainEvent):
	"""Emitted when a metric value crosses an alert rule threshold."""

	event_type: str = field(default="metric_threshold_breached", init=False)

	@classmethod
	def from_breach(
		cls,
		tenant_id: str,
		rule_id: str,
		metric_name: str,
		observed_value: float,
		threshold_value: float,
		operator: str,
	) -> "MetricThresholdBreached":
		return cls(
			tenant_id=tenant_id,
			actor_id="rule_engine",
			payload={
				"rule_id": rule_id,
				"metric_name": metric_name,
				"observed_value": observed_value,
				"threshold_value": threshold_value,
				"operator": operator,
			},
		)


@dataclass(frozen=True)
class AnomalyDetected(DomainEvent):
	"""Emitted when an anomaly is detected for a metric series."""

	event_type: str = field(default="anomaly_detected", init=False)

	@classmethod
	def from_detection(
		cls,
		tenant_id: str,
		metric_name: str,
		anomaly_id: str,
		anomaly_score: float,
		observed_value: float,
		expected_value: float,
		algorithm: str,
	) -> "AnomalyDetected":
		return cls(
			tenant_id=tenant_id,
			actor_id="anomaly_detector",
			payload={
				"anomaly_id": anomaly_id,
				"metric_name": metric_name,
				"anomaly_score": anomaly_score,
				"observed_value": observed_value,
				"expected_value": expected_value,
				"algorithm": algorithm,
			},
		)


# ─── Alert events ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AlertTriggered(DomainEvent):
	"""Emitted when a new alert is fired by the rule engine."""

	event_type: str = field(default="alert_triggered", init=False)

	@classmethod
	def from_alert(
		cls,
		tenant_id: str,
		alert_id: str,
		rule_id: str,
		severity: str,
		message: str,
	) -> "AlertTriggered":
		return cls(
			tenant_id=tenant_id,
			actor_id="rule_engine",
			payload={
				"alert_id": alert_id,
				"rule_id": rule_id,
				"severity": severity,
				"message": message,
			},
		)


@dataclass(frozen=True)
class AlertAcknowledged(DomainEvent):
	"""Emitted when an alert is acknowledged by an operator."""

	event_type: str = field(default="alert_acknowledged", init=False)

	@classmethod
	def from_ack(
		cls,
		tenant_id: str,
		alert_id: str,
		acknowledged_by: str,
	) -> "AlertAcknowledged":
		return cls(
			tenant_id=tenant_id,
			actor_id=acknowledged_by,
			payload={"alert_id": alert_id, "acknowledged_by": acknowledged_by},
		)


@dataclass(frozen=True)
class AlertResolved(DomainEvent):
	"""Emitted when an alert transitions to resolved."""

	event_type: str = field(default="alert_resolved", init=False)

	@classmethod
	def from_resolution(
		cls,
		tenant_id: str,
		alert_id: str,
		resolved_by: str,
		resolution_note: str = "",
	) -> "AlertResolved":
		return cls(
			tenant_id=tenant_id,
			actor_id=resolved_by,
			payload={
				"alert_id": alert_id,
				"resolved_by": resolved_by,
				"resolution_note": resolution_note,
			},
		)


@dataclass(frozen=True)
class AlertEscalated(DomainEvent):
	"""Emitted when an alert is escalated to the next level."""

	event_type: str = field(default="alert_escalated", init=False)

	@classmethod
	def from_escalation(
		cls,
		tenant_id: str,
		alert_id: str,
		escalation_level: int,
	) -> "AlertEscalated":
		return cls(
			tenant_id=tenant_id,
			actor_id="escalation_engine",
			payload={"alert_id": alert_id, "escalation_level": escalation_level},
		)


# ─── SLO events ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SLOBreached(DomainEvent):
	"""Emitted when an SLO compliance drops below its objective."""

	event_type: str = field(default="slo_breached", init=False)

	@classmethod
	def from_breach(
		cls,
		tenant_id: str,
		slo_id: str,
		service_name: str,
		current_compliance: float,
		objective_percent: float,
		error_budget_remaining_percent: float,
	) -> "SLOBreached":
		return cls(
			tenant_id=tenant_id,
			actor_id="slo_engine",
			payload={
				"slo_id": slo_id,
				"service_name": service_name,
				"current_compliance": current_compliance,
				"objective_percent": objective_percent,
				"error_budget_remaining_percent": error_budget_remaining_percent,
			},
		)


@dataclass(frozen=True)
class ErrorBudgetExhausted(DomainEvent):
	"""Emitted when the error budget reaches zero."""

	event_type: str = field(default="error_budget_exhausted", init=False)

	@classmethod
	def from_exhaustion(
		cls,
		tenant_id: str,
		slo_id: str,
		service_name: str,
		burn_rate: float,
	) -> "ErrorBudgetExhausted":
		return cls(
			tenant_id=tenant_id,
			actor_id="slo_engine",
			payload={
				"slo_id": slo_id,
				"service_name": service_name,
				"burn_rate": burn_rate,
			},
		)


# ─── Incident events ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IncidentRaised(DomainEvent):
	"""Emitted when a new incident is created."""

	event_type: str = field(default="incident_raised", init=False)

	@classmethod
	def from_incident(
		cls,
		tenant_id: str,
		incident_id: str,
		severity: str,
		title: str,
		owner: str | None,
	) -> "IncidentRaised":
		return cls(
			tenant_id=tenant_id,
			actor_id=owner or "system",
			payload={
				"incident_id": incident_id,
				"severity": severity,
				"title": title,
				"owner": owner,
			},
		)


@dataclass(frozen=True)
class IncidentResolved(DomainEvent):
	"""Emitted when an incident is resolved."""

	event_type: str = field(default="incident_resolved", init=False)

	@classmethod
	def from_resolution(
		cls,
		tenant_id: str,
		incident_id: str,
		resolved_by: str,
		duration_minutes: float,
	) -> "IncidentResolved":
		return cls(
			tenant_id=tenant_id,
			actor_id=resolved_by,
			payload={
				"incident_id": incident_id,
				"resolved_by": resolved_by,
				"duration_minutes": duration_minutes,
			},
		)


# ─── Health-check events ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class HealthCheckFailed(DomainEvent):
	"""Emitted when a health check probe returns an unhealthy result."""

	event_type: str = field(default="health_check_failed", init=False)

	@classmethod
	def from_failure(
		cls,
		tenant_id: str,
		check_id: str,
		service_name: str,
		endpoint: str,
		consecutive_failures: int,
	) -> "HealthCheckFailed":
		return cls(
			tenant_id=tenant_id,
			actor_id="health_checker",
			payload={
				"check_id": check_id,
				"service_name": service_name,
				"endpoint": endpoint,
				"consecutive_failures": consecutive_failures,
			},
		)


@dataclass(frozen=True)
class HealthCheckRecovered(DomainEvent):
	"""Emitted when a previously failing health check recovers."""

	event_type: str = field(default="health_check_recovered", init=False)

	@classmethod
	def from_recovery(
		cls,
		tenant_id: str,
		check_id: str,
		service_name: str,
	) -> "HealthCheckRecovered":
		return cls(
			tenant_id=tenant_id,
			actor_id="health_checker",
			payload={"check_id": check_id, "service_name": service_name},
		)


# ─── Dashboard events ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DashboardCreated(DomainEvent):
	"""Emitted when a new dashboard is created."""

	event_type: str = field(default="dashboard_created", init=False)

	@classmethod
	def from_dashboard(
		cls,
		tenant_id: str,
		dashboard_id: str,
		name: str,
		created_by: str,
	) -> "DashboardCreated":
		return cls(
			tenant_id=tenant_id,
			actor_id=created_by,
			payload={"dashboard_id": dashboard_id, "name": name},
		)


@dataclass(frozen=True)
class DashboardUpdated(DomainEvent):
	"""Emitted when a dashboard configuration is modified."""

	event_type: str = field(default="dashboard_updated", init=False)

	@classmethod
	def from_update(
		cls,
		tenant_id: str,
		dashboard_id: str,
		updated_by: str,
		widget_count: int,
	) -> "DashboardUpdated":
		return cls(
			tenant_id=tenant_id,
			actor_id=updated_by,
			payload={"dashboard_id": dashboard_id, "widget_count": widget_count},
		)


# ─── Trace events ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TraceCaptured(DomainEvent):
	"""Emitted when a distributed trace span is collected."""

	event_type: str = field(default="trace_captured", init=False)

	@classmethod
	def from_span(
		cls,
		tenant_id: str,
		trace_id: str,
		span_id: str,
		service_name: str,
		operation_name: str,
		duration_ms: float | None,
		error: bool,
	) -> "TraceCaptured":
		return cls(
			tenant_id=tenant_id,
			actor_id=service_name,
			payload={
				"trace_id": trace_id,
				"span_id": span_id,
				"service_name": service_name,
				"operation_name": operation_name,
				"duration_ms": duration_ms,
				"error": error,
			},
		)


# ─── Agent events ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AgentRegistered(DomainEvent):
	"""Emitted when a monitoring agent is registered."""

	event_type: str = field(default="agent_registered", init=False)

	@classmethod
	def from_agent(
		cls,
		tenant_id: str,
		agent_id: str,
		name: str,
		runtime: str,
		role: str,
		owner: str,
	) -> "AgentRegistered":
		return cls(
			tenant_id=tenant_id,
			actor_id=owner,
			payload={
				"agent_id": agent_id,
				"name": name,
				"runtime": runtime,
				"role": role,
				"owner": owner,
			},
		)


__all__ = [
	"DomainEvent",
	# metric
	"MetricRecorded",
	"MetricThresholdBreached",
	"AnomalyDetected",
	# alert
	"AlertTriggered",
	"AlertAcknowledged",
	"AlertResolved",
	"AlertEscalated",
	# SLO
	"SLOBreached",
	"ErrorBudgetExhausted",
	# incident
	"IncidentRaised",
	"IncidentResolved",
	# health check
	"HealthCheckFailed",
	"HealthCheckRecovered",
	# dashboard
	"DashboardCreated",
	"DashboardUpdated",
	# trace
	"TraceCaptured",
	# agent
	"AgentRegistered",
]
