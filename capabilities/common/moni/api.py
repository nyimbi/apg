"""APG Monitoring and Observability (MONI) — REST API Blueprint.

Provides a complete REST API surface for all MONI entities.
Mounted at /moni/api/v1 by the app factory.

All endpoints enforce tenant isolation via the X-Tenant-Id header or
the tenant_id query parameter (header takes precedence).

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from flask import Blueprint, jsonify, request

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .service import MoniService

blueprint = Blueprint("moni_api", __name__, url_prefix="/moni/api/v1")

# Module-level service instance — replaced in tests / app factory via set_service().
_service: MoniService | None = None


def set_service(svc: MoniService) -> None:
	"""Inject a MoniService instance (used by app factory and tests)."""
	global _service
	_service = svc


def _svc() -> MoniService:
	global _service
	if _service is None:
		_service = MoniService()
	return _service


def _tenant() -> str:
	"""Resolve tenant from header then query param, defaulting to 'default'."""
	return (
		request.headers.get("X-Tenant-Id")
		or request.args.get("tenant_id")
		or "default"
	)


def _ok(data: Any, status: int = 200):
	return jsonify({"ok": True, "data": data}), status


def _err(message: str, status: int = 400, code: str | None = None):
	return jsonify({"ok": False, "error": message, "code": code or "bad_request"}), status


def _paginate(rows: list[Any], page: int = 1, page_size: int = 50) -> dict[str, Any]:
	page = max(1, page)
	page_size = max(1, min(page_size, 500))
	start = (page - 1) * page_size
	end = start + page_size
	return {
		"items": rows[start:end],
		"total": len(rows),
		"page": page,
		"page_size": page_size,
		"pages": max(1, (len(rows) + page_size - 1) // page_size),
	}


# ─── Contract & health ───────────────────────────────────────────────────────

@blueprint.get("/contract")
def get_contract():
	"""Return the full capability contract for the tenant."""
	return _ok(get_capability_contract(_tenant()))


@blueprint.get("/health")
def health():
	"""Liveness probe — always 200 while the process is running."""
	svc = _svc()
	summary = svc.dashboard_summary(_tenant())
	return _ok({
		"status": "ok",
		"capability": "moni",
		"tenant_id": _tenant(),
		"summary": summary,
		"uptime_ms": int(time.time() * 1000),
	})


@blueprint.post("/evaluate")
def evaluate():
	"""Evaluate a raw context dict against the MONI rule engine."""
	ctx = request.get_json(force=True, silent=True) or {}
	return _ok(evaluate_capability_rules(ctx))


@blueprint.get("/dashboard")
def dashboard():
	"""Return dashboard KPIs and pending-review counts."""
	from .view_models import dashboard_model
	return _ok(dashboard_model(_svc(), _tenant()))


@blueprint.get("/pending-reviews")
def pending_reviews():
	"""Return all records in pending_review state for this tenant."""
	return _ok(_svc().list_pending_reviews(_tenant()))


# ─── Signal sources ───────────────────────────────────────────────────────────

@blueprint.get("/sources")
def list_sources():
	"""List all registered signal sources for the tenant."""
	rows = _svc().list_records(_tenant(), "sources")
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/sources")
def create_source():
	"""Register a new telemetry source."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
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
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/sources/<source_id>")
def get_source(source_id: str):
	"""Get a single signal source by ID."""
	key = f"{_tenant()}:{source_id}"
	record = _svc().sources.get(key)
	if not record:
		return _err(f"Source {source_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


@blueprint.delete("/sources/<source_id>")
def disable_source(source_id: str):
	"""Soft-disable a signal source (sets status=disabled)."""
	key = f"{_tenant()}:{source_id}"
	record = _svc().sources.get(key)
	if not record:
		return _err(f"Source {source_id!r} not found", 404, "not_found")
	record.status = "disabled"
	return _ok(asdict(record))


# ─── Signals ──────────────────────────────────────────────────────────────────

@blueprint.get("/signals")
def list_signals():
	"""List ingested signals for the tenant with optional type filter."""
	rows = _svc().list_records(_tenant(), "signals")
	signal_type = request.args.get("signal_type")
	if signal_type:
		rows = [r for r in rows if r.get("signal_type") == signal_type]
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/signals")
def ingest_signal():
	"""Ingest a telemetry signal (metric, log, or trace)."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
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
			cardinality=int(payload.get("cardinality", 0)),
			contains_pii=bool(payload.get("contains_pii", False)),
			pii_redacted=bool(payload.get("pii_redacted", True)),
			cardinality_exception_recorded=bool(payload.get("cardinality_exception_recorded", False)),
		)
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/signals/<signal_id>")
def get_signal(signal_id: str):
	"""Get a single signal by ID."""
	record = _svc().signals.get(signal_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Signal {signal_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


# ─── SLOs ─────────────────────────────────────────────────────────────────────

@blueprint.get("/slos")
def list_slos():
	"""List all SLO definitions for the tenant."""
	rows = _svc().list_records(_tenant(), "slos")
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/slos")
def create_slo():
	"""Create a new Service Level Objective."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().create_slo(
			tenant_id=payload.get("tenant_id", _tenant()),
			service_name=payload["service_name"],
			objective=payload["objective"],
			threshold=float(payload["threshold"]),
			window_minutes=int(payload["window_minutes"]),
			owner=payload["owner"],
			notification_route=payload.get("notification_route"),
		)
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/slos/<slo_id>")
def get_slo(slo_id: str):
	"""Get a single SLO by ID."""
	record = _svc().slos.get(slo_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"SLO {slo_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


@blueprint.delete("/slos/<slo_id>")
def retire_slo(slo_id: str):
	"""Retire an SLO (soft delete — sets status=retired)."""
	record = _svc().slos.get(slo_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"SLO {slo_id!r} not found", 404, "not_found")
	record.status = "retired"
	return _ok(asdict(record))


# ─── Alerts ───────────────────────────────────────────────────────────────────

@blueprint.get("/alerts")
def list_alerts():
	"""List alerts for the tenant with optional severity/status filters."""
	rows = _svc().list_records(_tenant(), "alerts")
	severity = request.args.get("severity")
	status = request.args.get("status")
	if severity:
		rows = [r for r in rows if r.get("severity") == severity]
	if status:
		rows = [r for r in rows if r.get("status") == status]
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/alerts")
def create_alert():
	"""Create a new alert. Critical alerts auto-open an incident."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().create_alert(
			tenant_id=payload.get("tenant_id", _tenant()),
			source_id=payload["source_id"],
			severity=payload["severity"],
			title=payload["title"],
			notification_route=payload.get("notification_route"),
			owner=payload.get("owner"),
		)
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/alerts/<alert_id>")
def get_alert(alert_id: str):
	"""Get a single alert by ID."""
	record = _svc().alerts.get(alert_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Alert {alert_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


@blueprint.post("/alerts/<alert_id>/acknowledge")
def acknowledge_alert(alert_id: str):
	"""Acknowledge an open alert."""
	from datetime import datetime
	record = _svc().alerts.get(alert_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Alert {alert_id!r} not found", 404, "not_found")
	if record.status not in ("open", "active"):
		return _err(f"Alert is already {record.status}", 409, "state_conflict")
	record.status = "acknowledged"
	record.acknowledged_at = datetime.utcnow()
	return _ok(asdict(record))


@blueprint.post("/alerts/<alert_id>/resolve")
def resolve_alert(alert_id: str):
	"""Resolve an open or acknowledged alert."""
	from datetime import datetime
	record = _svc().alerts.get(alert_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Alert {alert_id!r} not found", 404, "not_found")
	if record.status == "resolved":
		return _err("Alert is already resolved", 409, "state_conflict")
	record.status = "resolved"
	record.resolved_at = datetime.utcnow()
	return _ok(asdict(record))


@blueprint.delete("/alerts/<alert_id>")
def suppress_alert(alert_id: str):
	"""Suppress an alert (soft delete — sets status=suppressed)."""
	record = _svc().alerts.get(alert_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Alert {alert_id!r} not found", 404, "not_found")
	record.status = "suppressed"
	return _ok(asdict(record))


# ─── Incidents ────────────────────────────────────────────────────────────────

@blueprint.get("/incidents")
def list_incidents():
	"""List incidents for the tenant with optional severity/status filters."""
	rows = _svc().list_records(_tenant(), "incidents")
	severity = request.args.get("severity")
	status = request.args.get("status")
	if severity:
		rows = [r for r in rows if r.get("severity") == severity]
	if status:
		rows = [r for r in rows if r.get("status") == status]
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/incidents")
def create_incident():
	"""Create a new incident record."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().create_incident(
			tenant_id=payload.get("tenant_id", _tenant()),
			title=payload["title"],
			severity=payload["severity"],
			owner=payload.get("owner"),
			notification_route=payload.get("notification_route"),
			alert_ids=payload.get("alert_ids"),
		)
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/incidents/<incident_id>")
def get_incident(incident_id: str):
	"""Get a single incident by ID."""
	record = _svc().incidents.get(incident_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Incident {incident_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


@blueprint.post("/incidents/<incident_id>/resolve")
def resolve_incident(incident_id: str):
	"""Mark an incident as resolved."""
	from datetime import datetime
	record = _svc().incidents.get(incident_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Incident {incident_id!r} not found", 404, "not_found")
	record.status = "resolved"
	record.resolved_at = datetime.utcnow()
	return _ok(asdict(record))


@blueprint.get("/incidents/<incident_id>/timeline")
def incident_timeline(incident_id: str):
	"""Return audit events correlated to this incident."""
	record = _svc().incidents.get(incident_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Incident {incident_id!r} not found", 404, "not_found")
	events = [
		asdict(e)
		for e in _svc().audit_events
		if e.tenant_id == _tenant() and incident_id in str(e.details)
	]
	return _ok({"incident": asdict(record), "timeline": events})


# ─── Remediation ──────────────────────────────────────────────────────────────

@blueprint.get("/remediation")
def list_remediation():
	"""List remediation requests for the tenant."""
	rows = _svc().list_records(_tenant(), "remediation_requests")
	status = request.args.get("status")
	if status:
		rows = [r for r in rows if r.get("status") == status]
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/remediation")
def request_remediation():
	"""Request runbook-backed remediation for an incident."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
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
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")


@blueprint.get("/remediation/<request_id>")
def get_remediation_request(request_id: str):
	"""Get a single remediation request by ID."""
	record = _svc().remediation_requests.get(request_id)
	if not record or record.tenant_id != _tenant():
		return _err(f"Remediation request {request_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


@blueprint.post("/remediation/<request_id>/approve")
def approve_remediation(request_id: str):
	"""Approve a remediation request."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().decide_remediation(
			request_id=request_id,
			reviewer=payload["reviewer"],
			decision="approved",
			notes=payload.get("notes", ""),
		)
		return _ok(asdict(record))
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except (ValueError, KeyError) as exc:
		return _err(str(exc), 422, "validation_error")
	except PermissionError as exc:
		return _err(str(exc), 403, "policy_denied")


@blueprint.post("/remediation/<request_id>/reject")
def reject_remediation(request_id: str):
	"""Reject a remediation request."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().decide_remediation(
			request_id=request_id,
			reviewer=payload["reviewer"],
			decision="rejected",
			notes=payload.get("notes", ""),
		)
		return _ok(asdict(record))
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except (ValueError, KeyError) as exc:
		return _err(str(exc), 422, "validation_error")
	except PermissionError as exc:
		return _err(str(exc), 403, "policy_denied")


# ─── Monitoring agents ────────────────────────────────────────────────────────

@blueprint.get("/agents")
def list_agents():
	"""List registered monitoring agents for the tenant."""
	rows = _svc().list_records(_tenant(), "monitoring_agents")
	status = request.args.get("status")
	if status:
		rows = [r for r in rows if r.get("status") == status]
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/agents")
def register_agent():
	"""Register a first-class observability agent."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
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
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")
	except PermissionError as exc:
		return _err(str(exc), 403, "policy_denied")


@blueprint.get("/agents/<agent_id>")
def get_agent(agent_id: str):
	"""Get a single monitoring agent by ID."""
	key = f"{_tenant()}:{agent_id}"
	record = _svc().monitoring_agents.get(key)
	if not record:
		return _err(f"Agent {agent_id!r} not found", 404, "not_found")
	return _ok(asdict(record))


# ─── Lifecycle batches ────────────────────────────────────────────────────────

@blueprint.get("/lifecycle")
def list_lifecycle_batches():
	"""List lifecycle batch records for the tenant."""
	rows = _svc().list_records(_tenant(), "lifecycle_batches")
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


@blueprint.post("/lifecycle")
def validate_lifecycle_batch():
	"""Validate a MONI lifecycle batch — must declare Bytewax as the processor."""
	payload = request.get_json(force=True, silent=True) or {}
	try:
		record = _svc().validate_monitoring_lifecycle_batch(
			tenant_id=payload.get("tenant_id", _tenant()),
			event_stream=payload["event_stream"],
			mutation_count=int(payload["mutation_count"]),
		)
		return _ok(asdict(record), 201)
	except KeyError as exc:
		return _err(f"Missing required field: {exc}", 400, "missing_field")
	except ValueError as exc:
		return _err(str(exc), 422, "validation_error")
	except PermissionError as exc:
		return _err(str(exc), 403, "policy_denied")


# ─── Audit ────────────────────────────────────────────────────────────────────

@blueprint.get("/audit")
def list_audit():
	"""List audit events for the tenant, newest first."""
	rows = _svc().list_records(_tenant(), "audit_events")
	event_type = request.args.get("event_type")
	if event_type:
		rows = [r for r in rows if r.get("event_type") == event_type]
	rows = list(reversed(rows))
	page = int(request.args.get("page", 1))
	page_size = int(request.args.get("page_size", 50))
	return _ok(_paginate(rows, page, page_size))


# ─── Reports ──────────────────────────────────────────────────────────────────

@blueprint.get("/reports/summary")
def report_summary():
	"""High-level summary report across all MONI entities."""
	return _ok(_svc().dashboard_summary(_tenant()))


@blueprint.get("/reports/slo-compliance")
def report_slo_compliance():
	"""SLO compliance report for all active SLOs of the tenant."""
	slos = _svc().list_records(_tenant(), "slos")
	active = [s for s in slos if s.get("status") == "active"]
	breached = [s for s in active if s.get("status") == "breached"]
	return _ok({
		"total_slos": len(slos),
		"active_slos": len(active),
		"breached_slos": len(breached),
		"slos": active,
	})


@blueprint.get("/reports/alert-trends")
def report_alert_trends():
	"""Alert count grouped by severity for the tenant."""
	rows = _svc().list_records(_tenant(), "alerts")
	by_severity: dict[str, int] = {}
	by_status: dict[str, int] = {}
	for row in rows:
		sev = row.get("severity", "unknown")
		st = row.get("status", "unknown")
		by_severity[sev] = by_severity.get(sev, 0) + 1
		by_status[st] = by_status.get(st, 0) + 1
	return _ok({
		"total": len(rows),
		"by_severity": by_severity,
		"by_status": by_status,
	})


@blueprint.get("/reports/incident-mttr")
def report_incident_mttr():
	"""Mean time to resolution for resolved incidents."""
	from datetime import datetime
	rows = _svc().list_records(_tenant(), "incidents")
	resolved = [r for r in rows if r.get("status") == "resolved" and r.get("resolved_at")]
	if not resolved:
		return _ok({"mttr_minutes": None, "sample_size": 0})
	durations = []
	for r in resolved:
		try:
			created = datetime.fromisoformat(r["created_at"])
			resolved_at = datetime.fromisoformat(r["resolved_at"])
			durations.append((resolved_at - created).total_seconds() / 60)
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
	mttr = sum(durations) / len(durations) if durations else None
	return _ok({"mttr_minutes": mttr, "sample_size": len(durations)})
