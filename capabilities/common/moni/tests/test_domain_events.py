"""CI tests for MONI domain events.

Verifies every concrete event type can be constructed, serialised,
and carries correct payload keys.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import pytest
from datetime import datetime

from capabilities.common.moni.domain.events import (
	DomainEvent,
	MetricRecorded,
	MetricThresholdBreached,
	AnomalyDetected,
	AlertTriggered,
	AlertAcknowledged,
	AlertResolved,
	AlertEscalated,
	SLOBreached,
	ErrorBudgetExhausted,
	IncidentRaised,
	IncidentResolved,
	HealthCheckFailed,
	HealthCheckRecovered,
	DashboardCreated,
	DashboardUpdated,
	TraceCaptured,
	AgentRegistered,
)


TENANT = "tenant-events"
ACTOR = "test-actor"


# ─── DomainEvent base contract ────────────────────────────────────────────────

def test_domain_event_to_dict_includes_required_keys():
	ev = DomainEvent(
		event_type="test_event",
		tenant_id=TENANT,
		actor_id=ACTOR,
		payload={"k": "v"},
	)
	d = ev.to_dict()
	assert d["event_type"] == "test_event"
	assert d["tenant_id"] == TENANT
	assert d["actor_id"] == ACTOR
	assert d["capability_id"] == "moni"
	assert isinstance(d["timestamp"], str)  # ISO string
	assert d["payload"] == {"k": "v"}


def test_domain_event_is_frozen():
	"""Direct field assignment on a frozen dataclass must raise."""
	ev = DomainEvent(event_type="x", tenant_id="t", actor_id="a")
	with pytest.raises(AttributeError):  # FrozenInstanceError is a subclass
		ev.tenant_id = "other"  # type: ignore[misc]


def test_domain_event_default_timestamp_is_recent():
	before = datetime.utcnow()
	ev = DomainEvent(event_type="x", tenant_id="t", actor_id="a")
	after = datetime.utcnow()
	assert before <= ev.timestamp <= after


# ─── MetricRecorded ───────────────────────────────────────────────────────────

def test_metric_recorded_factory():
	ev = MetricRecorded.from_metric(
		tenant_id=TENANT,
		actor_id=ACTOR,
		metric_name="cpu.usage",
		value=78.5,
		source="host-1",
		labels={"env": "prod"},
	)
	assert ev.event_type == "metric_recorded"
	assert ev.tenant_id == TENANT
	assert ev.payload["metric_name"] == "cpu.usage"
	assert ev.payload["value"] == 78.5
	assert ev.payload["labels"] == {"env": "prod"}


def test_metric_recorded_default_empty_labels():
	ev = MetricRecorded.from_metric(TENANT, ACTOR, "m", 1.0, "src")
	assert ev.payload["labels"] == {}


# ─── MetricThresholdBreached ──────────────────────────────────────────────────

def test_metric_threshold_breached_factory():
	ev = MetricThresholdBreached.from_breach(
		tenant_id=TENANT,
		rule_id="rule-1",
		metric_name="cpu.usage",
		observed_value=92.0,
		threshold_value=90.0,
		operator="gt",
	)
	assert ev.event_type == "metric_threshold_breached"
	assert ev.payload["observed_value"] == 92.0
	assert ev.payload["threshold_value"] == 90.0
	assert ev.payload["operator"] == "gt"


# ─── AnomalyDetected ──────────────────────────────────────────────────────────

def test_anomaly_detected_factory():
	ev = AnomalyDetected.from_detection(
		tenant_id=TENANT,
		metric_name="latency",
		anomaly_id="anom-1",
		anomaly_score=0.95,
		observed_value=500.0,
		expected_value=120.0,
		algorithm="z_score",
	)
	assert ev.event_type == "anomaly_detected"
	assert ev.payload["anomaly_score"] == 0.95
	assert ev.payload["algorithm"] == "z_score"


# ─── AlertTriggered ───────────────────────────────────────────────────────────

def test_alert_triggered_factory():
	ev = AlertTriggered.from_alert(
		tenant_id=TENANT,
		alert_id="alert-1",
		rule_id="rule-1",
		severity="critical",
		message="CPU too high",
	)
	assert ev.event_type == "alert_triggered"
	assert ev.payload["severity"] == "critical"
	assert ev.actor_id == "rule_engine"


# ─── AlertAcknowledged ────────────────────────────────────────────────────────

def test_alert_acknowledged_factory():
	ev = AlertAcknowledged.from_ack(TENANT, "alert-1", "sre-lead")
	assert ev.event_type == "alert_acknowledged"
	assert ev.actor_id == "sre-lead"
	assert ev.payload["alert_id"] == "alert-1"


# ─── AlertResolved ────────────────────────────────────────────────────────────

def test_alert_resolved_factory():
	ev = AlertResolved.from_resolution(TENANT, "alert-1", "sre", "restarted pod")
	assert ev.event_type == "alert_resolved"
	assert ev.payload["resolution_note"] == "restarted pod"


def test_alert_resolved_default_empty_note():
	ev = AlertResolved.from_resolution(TENANT, "alert-1", "sre")
	assert ev.payload["resolution_note"] == ""


# ─── AlertEscalated ───────────────────────────────────────────────────────────

def test_alert_escalated_factory():
	ev = AlertEscalated.from_escalation(TENANT, "alert-1", 2)
	assert ev.event_type == "alert_escalated"
	assert ev.payload["escalation_level"] == 2


# ─── SLOBreached ─────────────────────────────────────────────────────────────

def test_slo_breached_factory():
	ev = SLOBreached.from_breach(
		tenant_id=TENANT,
		slo_id="slo-1",
		service_name="orders",
		current_compliance=99.5,
		objective_percent=99.9,
		error_budget_remaining_percent=30.0,
	)
	assert ev.event_type == "slo_breached"
	assert ev.payload["current_compliance"] == 99.5
	assert ev.payload["error_budget_remaining_percent"] == 30.0


# ─── ErrorBudgetExhausted ─────────────────────────────────────────────────────

def test_error_budget_exhausted_factory():
	ev = ErrorBudgetExhausted.from_exhaustion(TENANT, "slo-1", "orders", 14.4)
	assert ev.event_type == "error_budget_exhausted"
	assert ev.payload["burn_rate"] == 14.4


# ─── IncidentRaised ───────────────────────────────────────────────────────────

def test_incident_raised_factory():
	ev = IncidentRaised.from_incident(
		tenant_id=TENANT,
		incident_id="inc-1",
		severity="critical",
		title="DB unreachable",
		owner="sre-lead",
	)
	assert ev.event_type == "incident_raised"
	assert ev.payload["severity"] == "critical"
	assert ev.actor_id == "sre-lead"


def test_incident_raised_no_owner_defaults_system():
	ev = IncidentRaised.from_incident(TENANT, "inc-1", "high", "Issue", None)
	assert ev.actor_id == "system"


# ─── IncidentResolved ─────────────────────────────────────────────────────────

def test_incident_resolved_factory():
	ev = IncidentResolved.from_resolution(TENANT, "inc-1", "sre", 45.5)
	assert ev.event_type == "incident_resolved"
	assert ev.payload["duration_minutes"] == 45.5


# ─── HealthCheckFailed ────────────────────────────────────────────────────────

def test_health_check_failed_factory():
	ev = HealthCheckFailed.from_failure(
		tenant_id=TENANT,
		check_id="hc-1",
		service_name="payments",
		endpoint="https://pay.svc/health",
		consecutive_failures=3,
	)
	assert ev.event_type == "health_check_failed"
	assert ev.payload["consecutive_failures"] == 3
	assert ev.actor_id == "health_checker"


# ─── HealthCheckRecovered ─────────────────────────────────────────────────────

def test_health_check_recovered_factory():
	ev = HealthCheckRecovered.from_recovery(TENANT, "hc-1", "payments")
	assert ev.event_type == "health_check_recovered"
	assert ev.payload["service_name"] == "payments"


# ─── DashboardCreated ─────────────────────────────────────────────────────────

def test_dashboard_created_factory():
	ev = DashboardCreated.from_dashboard(TENANT, "dash-1", "Ops Overview", "admin")
	assert ev.event_type == "dashboard_created"
	assert ev.payload["name"] == "Ops Overview"


# ─── DashboardUpdated ─────────────────────────────────────────────────────────

def test_dashboard_updated_factory():
	ev = DashboardUpdated.from_update(TENANT, "dash-1", "admin", 5)
	assert ev.event_type == "dashboard_updated"
	assert ev.payload["widget_count"] == 5


# ─── TraceCaptured ────────────────────────────────────────────────────────────

def test_trace_captured_factory():
	ev = TraceCaptured.from_span(
		tenant_id=TENANT,
		trace_id="trace-123",
		span_id="span-456",
		service_name="orders",
		operation_name="POST /orders",
		duration_ms=42.5,
		error=False,
	)
	assert ev.event_type == "trace_captured"
	assert ev.payload["trace_id"] == "trace-123"
	assert ev.payload["duration_ms"] == 42.5
	assert ev.payload["error"] is False


def test_trace_captured_with_error():
	ev = TraceCaptured.from_span(TENANT, "t", "s", "svc", "op", None, True)
	assert ev.payload["error"] is True
	assert ev.payload["duration_ms"] is None


# ─── AgentRegistered ─────────────────────────────────────────────────────────

def test_agent_registered_factory():
	ev = AgentRegistered.from_agent(
		tenant_id=TENANT,
		agent_id="agent-1",
		name="SLO Watcher",
		runtime="codex",
		role="slo_reviewer",
		owner="platform",
	)
	assert ev.event_type == "agent_registered"
	assert ev.payload["runtime"] == "codex"
	assert ev.actor_id == "platform"


# ─── to_dict round-trip for all events ───────────────────────────────────────

def test_all_events_serialise_to_dict():
	events = [
		MetricRecorded.from_metric(TENANT, ACTOR, "m", 1.0, "src"),
		MetricThresholdBreached.from_breach(TENANT, "r1", "m", 95.0, 90.0, "gt"),
		AnomalyDetected.from_detection(TENANT, "m", "a1", 0.9, 200.0, 100.0, "z_score"),
		AlertTriggered.from_alert(TENANT, "al1", "r1", "high", "msg"),
		AlertAcknowledged.from_ack(TENANT, "al1", "sre"),
		AlertResolved.from_resolution(TENANT, "al1", "sre"),
		AlertEscalated.from_escalation(TENANT, "al1", 1),
		SLOBreached.from_breach(TENANT, "s1", "svc", 99.5, 99.9, 30.0),
		ErrorBudgetExhausted.from_exhaustion(TENANT, "s1", "svc", 14.4),
		IncidentRaised.from_incident(TENANT, "i1", "critical", "Title", "owner"),
		IncidentResolved.from_resolution(TENANT, "i1", "sre", 60.0),
		HealthCheckFailed.from_failure(TENANT, "hc1", "svc", "url", 3),
		HealthCheckRecovered.from_recovery(TENANT, "hc1", "svc"),
		DashboardCreated.from_dashboard(TENANT, "d1", "Name", "user"),
		DashboardUpdated.from_update(TENANT, "d1", "user", 3),
		TraceCaptured.from_span(TENANT, "t1", "s1", "svc", "op", 10.0, False),
		AgentRegistered.from_agent(TENANT, "ag1", "Bot", "codex", "slo_reviewer", "owner"),
	]
	for ev in events:
		d = ev.to_dict()
		assert d["capability_id"] == "moni"
		assert d["tenant_id"] == TENANT
		assert "event_type" in d
		assert "timestamp" in d
		assert "payload" in d
