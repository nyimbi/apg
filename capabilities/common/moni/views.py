"""APG Monitoring and Observability (MONI) — Flask Blueprint UI views.

Plain Flask Blueprint — no flask_appbuilder dependency.

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, redirect, request, url_for

from .capability_contract import get_capability_contract
from .service import MoniService
from .view_models import (
	adapter_health_model,
	alert_center_model,
	analytics_model,
	audit_timeline_model,
	dashboard_model,
	incident_model,
	lifecycle_batch_model,
	monitoring_agent_roster_model,
	remediation_model,
	settings_model,
	signal_explorer_model,
	slo_model,
	source_inventory_model,
)

blueprint = Blueprint("moni_views", __name__, url_prefix="/moni")

# Module-level default service instance (replaced in tests / app factory).
_service: MoniService | None = None


def _svc() -> MoniService:
	global _service
	if _service is None:
		_service = MoniService()
	return _service


def _tenant() -> str:
	return request.args.get("tenant_id", "default")


# ─── Dashboard ────────────────────────────────────────────────────────────────

@blueprint.get("/dashboard")
def list_dashboard():
	"""Overview dashboard — summary KPIs and pending reviews."""
	model = dashboard_model(_svc(), _tenant())
	return jsonify(model)


# ─── Sources ──────────────────────────────────────────────────────────────────

@blueprint.get("/sources")
def list_sources():
	model = source_inventory_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/sources")
def create_source():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().register_source(
		tenant_id=payload.get("tenant_id", _tenant()),
		source_id=payload["source_id"],
		service_name=payload["service_name"],
		environment=payload["environment"],
		owner=payload["owner"],
		allowed_signal_types=payload.get("allowed_signal_types"),
		notification_route=payload.get("notification_route"),
		status=payload.get("status", "active"),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Signals / Metrics / Logs / Traces ───────────────────────────────────────

@blueprint.get("/signals")
def list_signals():
	model = signal_explorer_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/signals")
def ingest_signal():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().ingest_signal(
		tenant_id=payload.get("tenant_id", _tenant()),
		source_id=payload["source_id"],
		signal_type=payload["signal_type"],
		name=payload["name"],
		value=payload.get("value"),
		labels=payload.get("labels"),
		severity=payload.get("severity", "info"),
		trace_id=payload.get("trace_id"),
		service_name=payload.get("service_name"),
		cardinality=payload.get("cardinality", 0),
		contains_pii=payload.get("contains_pii", False),
		pii_redacted=payload.get("pii_redacted", True),
		cardinality_exception_recorded=payload.get("cardinality_exception_recorded", False),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Alerts ───────────────────────────────────────────────────────────────────

@blueprint.get("/alerts")
def list_alerts():
	model = alert_center_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/alerts")
def create_alert():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().create_alert(
		tenant_id=payload.get("tenant_id", _tenant()),
		source_id=payload["source_id"],
		severity=payload["severity"],
		title=payload["title"],
		notification_route=payload.get("notification_route"),
		owner=payload.get("owner"),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


@blueprint.post("/alerts/<alert_id>/acknowledge")
def acknowledge_alert(alert_id: str):
	alerts = _svc().alerts
	alert = alerts.get(alert_id)
	if not alert:
		return jsonify({"error": "not_found", "alert_id": alert_id}), 404
	if alert.tenant_id != _tenant():
		return jsonify({"error": "forbidden"}), 403
	alert.status = "acknowledged"
	alert.acknowledged_at = datetime.utcnow()
	from dataclasses import asdict
	return jsonify(asdict(alert))


@blueprint.post("/alerts/<alert_id>/resolve")
def resolve_alert(alert_id: str):
	alerts = _svc().alerts
	alert = alerts.get(alert_id)
	if not alert:
		return jsonify({"error": "not_found", "alert_id": alert_id}), 404
	if alert.tenant_id != _tenant():
		return jsonify({"error": "forbidden"}), 403
	alert.status = "resolved"
	alert.resolved_at = datetime.utcnow()
	from dataclasses import asdict
	return jsonify(asdict(alert))


# ─── SLOs ─────────────────────────────────────────────────────────────────────

@blueprint.get("/slos")
def list_slos():
	model = slo_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/slos")
def create_slo():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().create_slo(
		tenant_id=payload.get("tenant_id", _tenant()),
		service_name=payload["service_name"],
		objective=payload["objective"],
		threshold=float(payload["threshold"]),
		window_minutes=int(payload["window_minutes"]),
		owner=payload["owner"],
		notification_route=payload.get("notification_route"),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Incidents ────────────────────────────────────────────────────────────────

@blueprint.get("/incidents")
def list_incidents():
	model = incident_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/incidents")
def create_incident():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().create_incident(
		tenant_id=payload.get("tenant_id", _tenant()),
		title=payload["title"],
		severity=payload["severity"],
		owner=payload.get("owner"),
		notification_route=payload.get("notification_route"),
		alert_ids=payload.get("alert_ids"),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Remediation ──────────────────────────────────────────────────────────────

@blueprint.get("/remediation")
def list_remediation():
	model = remediation_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/remediation")
def request_remediation():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().request_remediation(
		tenant_id=payload.get("tenant_id", _tenant()),
		incident_id=payload["incident_id"],
		requester=payload["requester"],
		environment=payload["environment"],
		runbook_id=payload["runbook_id"],
		runbook_approved=bool(payload.get("runbook_approved", False)),
		proposed_action=payload["proposed_action"],
		reason=payload["reason"],
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


@blueprint.post("/remediation/<request_id>/decide")
def decide_remediation(request_id: str):
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().decide_remediation(
		request_id=request_id,
		reviewer=payload["reviewer"],
		decision=payload["decision"],
		notes=payload.get("notes", ""),
	)
	from dataclasses import asdict
	return jsonify(asdict(record))


# ─── Analytics ────────────────────────────────────────────────────────────────

@blueprint.get("/analytics")
def get_analytics():
	model = analytics_model(_svc(), _tenant())
	return jsonify(model)


# ─── Adapters ─────────────────────────────────────────────────────────────────

@blueprint.get("/adapters")
def get_adapter_health():
	return jsonify(adapter_health_model(_tenant()))


# ─── Agents ───────────────────────────────────────────────────────────────────

@blueprint.get("/agents")
def list_agents():
	model = monitoring_agent_roster_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/agents")
def register_agent():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().register_monitoring_agent(
		tenant_id=payload.get("tenant_id", _tenant()),
		agent_id=payload["agent_id"],
		name=payload["name"],
		runtime=payload["runtime"],
		role=payload["role"],
		scope=payload["scope"],
		owner=payload["owner"],
		purpose=payload["purpose"],
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Lifecycle batches ────────────────────────────────────────────────────────

@blueprint.get("/lifecycle")
def list_lifecycle_batches():
	model = lifecycle_batch_model(_svc(), _tenant())
	return jsonify(model)


@blueprint.post("/lifecycle")
def validate_lifecycle_batch():
	payload = request.get_json(force=True, silent=True) or {}
	record = _svc().validate_monitoring_lifecycle_batch(
		tenant_id=payload.get("tenant_id", _tenant()),
		event_stream=payload["event_stream"],
		mutation_count=int(payload["mutation_count"]),
	)
	from dataclasses import asdict
	return jsonify(asdict(record)), 201


# ─── Audit ────────────────────────────────────────────────────────────────────

@blueprint.get("/audit")
def list_audit():
	model = audit_timeline_model(_svc(), _tenant())
	return jsonify(model)


# ─── Settings ─────────────────────────────────────────────────────────────────

@blueprint.get("/settings")
def get_settings():
	return jsonify(settings_model(_tenant()))


# ─── Health ───────────────────────────────────────────────────────────────────

@blueprint.get("/health")
def health():
	summary = _svc().dashboard_summary(_tenant())
	return jsonify({"status": "ok", "capability": "moni", "summary": summary})


# ─── Pending reviews ─────────────────────────────────────────────────────────

@blueprint.get("/reviews/pending")
def list_pending_reviews():
	return jsonify(_svc().list_pending_reviews(_tenant()))
