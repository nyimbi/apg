"""Flask Blueprint REST API for APG Clinical Analytics."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from flask import Blueprint, jsonify, request

from .models import (
	AnalyticsReportCreate,
	CareGapCreate,
	CohortCreate,
	CohortUpdate,
	MetricRecordCreate,
	PredictionModelCreate,
	QualityIndicatorCreate,
)
from .service import ClinicalAnalyticsService, PolicyViolationError

bp = Blueprint("healthcare_ana", __name__, url_prefix="/api/healthcare/ana")
_svc = ClinicalAnalyticsService()


def _run(coro: Any) -> Any:
	loop = asyncio.new_event_loop()
	try:
		return asyncio.run(coro)
	finally:
		loop.close()


def _err(msg: str, status: int = 400) -> Any:
	return jsonify({"error": msg}), status


def _tenant() -> str:
	return request.headers.get("X-Tenant-ID", request.args.get("tenant_id", "default"))


# ── contract ──────────────────────────────────────────────────────────────────

@bp.get("/contract")
def get_contract():
	"""GET /api/healthcare/ana/contract — return capability contract."""
	return jsonify(_run(_svc.describe(_tenant())))


# ── cohorts ───────────────────────────────────────────────────────────────────

@bp.get("/cohorts")
def list_cohorts():
	"""GET /api/healthcare/ana/cohorts — list cohorts for tenant."""
	tid = _tenant()
	segment = request.args.get("segment")
	status = request.args.get("status")
	cohorts = _run(_svc.list_cohorts(tid, segment=segment, status=status))
	return jsonify({"items": [c.model_dump(mode="json") for c in cohorts], "count": len(cohorts)})


@bp.post("/cohorts")
def create_cohort():
	"""POST /api/healthcare/ana/cohorts — create a new cohort."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		payload = CohortCreate(**data)
		cohort = _run(_svc.create_cohort(payload))
		return jsonify(cohort.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.get("/cohorts/<cohort_id>")
def get_cohort(cohort_id: str):
	"""GET /api/healthcare/ana/cohorts/<id> — get cohort detail."""
	cohort = _run(_svc.get_cohort(_tenant(), cohort_id))
	if cohort is None:
		return _err("cohort_not_found", 404)
	return jsonify(cohort.model_dump(mode="json"))


@bp.put("/cohorts/<cohort_id>")
def update_cohort(cohort_id: str):
	"""PUT /api/healthcare/ana/cohorts/<id> — update cohort."""
	data = request.get_json(silent=True) or {}
	try:
		payload = CohortUpdate(**data)
		cohort = _run(_svc.update_cohort(_tenant(), cohort_id, payload))
		if cohort is None:
			return _err("cohort_not_found", 404)
		return jsonify(cohort.model_dump(mode="json"))
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.post("/cohorts/<cohort_id>/activate")
def activate_cohort(cohort_id: str):
	"""POST /api/healthcare/ana/cohorts/<id>/activate."""
	cohort = _run(_svc.activate_cohort(_tenant(), cohort_id))
	if cohort is None:
		return _err("cohort_not_found", 404)
	return jsonify(cohort.model_dump(mode="json"))


@bp.delete("/cohorts/<cohort_id>")
def delete_cohort(cohort_id: str):
	"""DELETE /api/healthcare/ana/cohorts/<id>."""
	try:
		deleted = _run(_svc.delete_cohort(_tenant(), cohort_id))
		if not deleted:
			return _err("cohort_not_found", 404)
		return jsonify({"deleted": True, "id": cohort_id})
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── metrics ───────────────────────────────────────────────────────────────────

@bp.get("/metrics")
def list_metrics():
	"""GET /api/healthcare/ana/metrics."""
	tid = _tenant()
	metric_type = request.args.get("metric_type")
	cohort_id = request.args.get("cohort_id")
	period = request.args.get("period")
	metrics = _run(_svc.list_metrics(tid, metric_type=metric_type, cohort_id=cohort_id, period=period))
	return jsonify({"items": [m.model_dump(mode="json") for m in metrics], "count": len(metrics)})


@bp.post("/metrics")
def record_metric():
	"""POST /api/healthcare/ana/metrics."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	# Coerce datetime strings
	for field in ("period_start", "period_end"):
		if field in data and isinstance(data[field], str):
			data[field] = datetime.fromisoformat(data[field])
	try:
		payload = MetricRecordCreate(**data)
		rec = _run(_svc.record_metric(payload))
		return jsonify(rec.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


# ── prediction models ─────────────────────────────────────────────────────────

@bp.get("/models")
def list_models():
	"""GET /api/healthcare/ana/models."""
	models = _run(_svc.list_prediction_models(_tenant()))
	return jsonify({"items": [m.model_dump(mode="json") for m in models], "count": len(models)})


@bp.post("/models")
def create_model():
	"""POST /api/healthcare/ana/models."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		payload = PredictionModelCreate(**data)
		model = _run(_svc.create_prediction_model(payload))
		return jsonify(model.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.post("/models/<model_id>/predict")
def generate_prediction(model_id: str):
	"""POST /api/healthcare/ana/models/<id>/predict."""
	data = request.get_json(silent=True) or {}
	try:
		result = _run(_svc.generate_prediction(_tenant(), model_id, data))
		return jsonify(result)
	except ValueError as e:
		return _err(str(e), 404)
	except PolicyViolationError as e:
		return _err(str(e), 403)


# ── quality indicators ────────────────────────────────────────────────────────

@bp.get("/quality-indicators")
def list_quality_indicators():
	"""GET /api/healthcare/ana/quality-indicators."""
	period = request.args.get("period")
	indicators = _run(_svc.list_quality_indicators(_tenant(), period=period))
	return jsonify({"items": [qi.model_dump(mode="json") for qi in indicators], "count": len(indicators)})


@bp.post("/quality-indicators")
def record_quality_indicator():
	"""POST /api/healthcare/ana/quality-indicators."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		payload = QualityIndicatorCreate(**data)
		qi = _run(_svc.record_quality_indicator(payload))
		return jsonify(qi.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


# ── care gaps ─────────────────────────────────────────────────────────────────

@bp.get("/care-gaps")
def list_care_gaps():
	"""GET /api/healthcare/ana/care-gaps."""
	tid = _tenant()
	patient_id = request.args.get("patient_id")
	severity = request.args.get("severity")
	status = request.args.get("status")
	gaps = _run(_svc.list_care_gaps(tid, patient_id=patient_id, severity=severity, status=status))
	return jsonify({"items": [g.model_dump(mode="json") for g in gaps], "count": len(gaps)})


@bp.post("/care-gaps")
def identify_care_gap():
	"""POST /api/healthcare/ana/care-gaps."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	try:
		payload = CareGapCreate(**data)
		gap = _run(_svc.identify_care_gap(payload))
		return jsonify(gap.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.post("/care-gaps/<gap_id>/resolve")
def resolve_care_gap(gap_id: str):
	"""POST /api/healthcare/ana/care-gaps/<id>/resolve."""
	gap = _run(_svc.resolve_care_gap(_tenant(), gap_id))
	if gap is None:
		return _err("care_gap_not_found", 404)
	return jsonify(gap.model_dump(mode="json"))


# ── reports ───────────────────────────────────────────────────────────────────

@bp.get("/reports")
def list_reports():
	"""GET /api/healthcare/ana/reports."""
	report_type = request.args.get("report_type")
	reports = _run(_svc.list_reports(_tenant(), report_type=report_type))
	return jsonify({"items": [r.model_dump(mode="json") for r in reports], "count": len(reports)})


@bp.post("/reports")
def generate_report():
	"""POST /api/healthcare/ana/reports."""
	data = request.get_json(silent=True) or {}
	data.setdefault("tenant_id", _tenant())
	for field in ("period_start", "period_end"):
		if field in data and isinstance(data[field], str):
			data[field] = datetime.fromisoformat(data[field])
	try:
		payload = AnalyticsReportCreate(**data)
		report = _run(_svc.generate_report(payload))
		return jsonify(report.model_dump(mode="json")), 201
	except PolicyViolationError as e:
		return _err(str(e), 403)
	except Exception as e:
		return _err(str(e))


@bp.get("/reports/<report_id>")
def get_report(report_id: str):
	"""GET /api/healthcare/ana/reports/<id>."""
	report = _run(_svc.get_report(_tenant(), report_id))
	if report is None:
		return _err("report_not_found", 404)
	return jsonify(report.model_dump(mode="json"))


# ── dashboard ─────────────────────────────────────────────────────────────────

@bp.get("/dashboard")
def dashboard():
	"""GET /api/healthcare/ana/dashboard."""
	return jsonify(_run(_svc.dashboard_summary(_tenant())))
