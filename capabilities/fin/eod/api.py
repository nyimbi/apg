"""REST API for APG EOD/BOD Processing Engine.

URL prefix: /api/fin/eod
All endpoints are synchronous Flask wrappers around the async EODService.
Use asyncio.run() or an ASGI adapter in production.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import date
from functools import wraps
from typing import Any

from flask import Blueprint, Response, jsonify, request

try:
	from .service import EODService
	from .models import EODJobType
except ImportError:  # pragma: no cover
	from service import EODService  # type: ignore
	from models import EODJobType   # type: ignore

_log = logging.getLogger(__name__)

bp = Blueprint("fin_eod", __name__, url_prefix="/api/fin/eod")
_SERVICE = EODService()


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask handler."""
	try:
		loop = asyncio.get_event_loop()
		if loop.is_running():
			import concurrent.futures
			with concurrent.futures.ThreadPoolExecutor() as pool:
				future = pool.submit(asyncio.run, coro)
				return future.result()
		return loop.run_until_complete(coro)
	except RuntimeError:
		return asyncio.run(coro)


def _ok(data: Any, status: int = 200) -> Response:
	if hasattr(data, "model_dump"):
		data = data.model_dump()
	return jsonify({"ok": True, "data": data}), status


def _err(msg: str, status: int = 400) -> Response:
	return jsonify({"ok": False, "error": msg}), status


def _tenant() -> str | None:
	return request.headers.get("X-Tenant-Id") or request.args.get("tenant_id")


def _require_tenant(fn: Any) -> Any:
	@wraps(fn)
	def wrapper(*args: Any, **kwargs: Any) -> Any:
		tid = _tenant()
		if not tid:
			return _err("X-Tenant-Id header or tenant_id query param required", 401)
		return fn(*args, tenant_id=tid, **kwargs)
	return wrapper


# ── Run endpoints ─────────────────────────────────────────────────────────────

@bp.post("/run")
@_require_tenant
def run_eod(tenant_id: str) -> Response:
	"""POST /api/fin/eod/run
	Body: {"eod_date": "2026-06-11", "dry_run": false}
	"""
	body    = request.get_json(silent=True) or {}
	eod_date = body.get("eod_date", date.today().isoformat())
	dry_run  = bool(body.get("dry_run", False))
	try:
		result = _run(_SERVICE.run_eod(tenant_id, eod_date, dry_run))
		return _ok(result)
	except Exception as ex:
		_log.exception("run_eod failed")
		return _err(str(ex), 500)


@bp.post("/bod")
@_require_tenant
def run_bod(tenant_id: str) -> Response:
	"""POST /api/fin/eod/bod
	Body: {"bod_date": "2026-06-11"}
	"""
	body     = request.get_json(silent=True) or {}
	bod_date = body.get("bod_date", date.today().isoformat())
	try:
		result = _run(_SERVICE.run_bod(tenant_id, bod_date))
		return _ok(result)
	except Exception as ex:
		_log.exception("run_bod failed")
		return _err(str(ex), 500)


@bp.post("/jobs/<job_name>")
@_require_tenant
def run_job(tenant_id: str, job_name: str) -> Response:
	"""POST /api/fin/eod/jobs/{job_name}
	Body: {"processing_date": "2026-06-11", "dry_run": false}
	"""
	body = request.get_json(silent=True) or {}
	processing_date = body.get("processing_date", date.today().isoformat())
	dry_run         = bool(body.get("dry_run", False))
	try:
		result = _run(_SERVICE.run_job(tenant_id, job_name, processing_date, dry_run))
		return _ok(result)
	except ValueError as ex:
		return _err(str(ex), 400)
	except Exception as ex:
		_log.exception("run_job failed")
		return _err(str(ex), 500)


# ── Status & query ────────────────────────────────────────────────────────────

@bp.get("/status/<processing_date>")
@_require_tenant
def get_eod_status(tenant_id: str, processing_date: str) -> Response:
	"""GET /api/fin/eod/status/2026-06-11"""
	try:
		data = _run(_SERVICE.get_eod_status(tenant_id, processing_date))
		return _ok(data)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/jobs/<processing_date>/<job_name>")
@_require_tenant
def get_job_result(tenant_id: str, processing_date: str, job_name: str) -> Response:
	"""GET /api/fin/eod/jobs/2026-06-11/interest_accrual_batch"""
	result = _run(_SERVICE.get_job_result(tenant_id, processing_date, job_name))
	if result is None:
		return _err(f"Job result not found for {job_name} on {processing_date}", 404)
	return _ok(result)


@bp.post("/jobs/<processing_date>/<job_name>/retry")
@_require_tenant
def retry_failed_job(tenant_id: str, processing_date: str, job_name: str) -> Response:
	"""POST /api/fin/eod/jobs/2026-06-11/interest_accrual_batch/retry"""
	try:
		result = _run(_SERVICE.retry_failed_job(tenant_id, processing_date, job_name))
		return _ok(result)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/history")
@_require_tenant
def get_eod_history(tenant_id: str) -> Response:
	"""GET /api/fin/eod/history?from_date=2026-05-01&to_date=2026-06-11"""
	from_date = request.args.get("from_date", "2026-01-01")
	to_date   = request.args.get("to_date",   date.today().isoformat())
	try:
		data = _run(_SERVICE.get_eod_history(tenant_id, from_date, to_date))
		return _ok(data)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/exceptions/<processing_date>")
@_require_tenant
def get_processing_exceptions(tenant_id: str, processing_date: str) -> Response:
	"""GET /api/fin/eod/exceptions/2026-06-11"""
	items = _run(_SERVICE.get_processing_exceptions(tenant_id, processing_date))
	return _ok([e.model_dump() for e in items])


@bp.post("/exceptions/<exception_id>/resolve")
@_require_tenant
def resolve_exception(tenant_id: str, exception_id: str) -> Response:
	"""POST /api/fin/eod/exceptions/{exception_id}/resolve
	Body: {"resolution": "Corrected GL entry", "resolved_by": "user@example.com"}
	"""
	body        = request.get_json(silent=True) or {}
	resolution  = body.get("resolution", "")
	resolved_by = body.get("resolved_by", "unknown")
	if not resolution:
		return _err("resolution is required", 400)
	try:
		exc = _run(_SERVICE.resolve_exception(tenant_id, exception_id, resolution, resolved_by))
		return _ok(exc)
	except KeyError as ex:
		return _err(str(ex), 404)
	except PermissionError as ex:
		return _err(str(ex), 403)


@bp.get("/pending")
@_require_tenant
def get_pending_items(tenant_id: str) -> Response:
	"""GET /api/fin/eod/pending"""
	data = _run(_SERVICE.get_pending_items(tenant_id))
	return _ok(data)


@bp.post("/schedule")
@_require_tenant
def schedule_job(tenant_id: str) -> Response:
	"""POST /api/fin/eod/schedule
	Body: {"job_name": "interest_accrual_batch", "scheduled_time": "2026-06-12T22:00:00Z", "parameters": {}}
	"""
	body           = request.get_json(silent=True) or {}
	job_name       = body.get("job_name", "")
	scheduled_time = body.get("scheduled_time", "")
	parameters     = body.get("parameters", {})
	if not job_name or not scheduled_time:
		return _err("job_name and scheduled_time are required", 400)
	try:
		sched = _run(_SERVICE.schedule_job(tenant_id, job_name, scheduled_time, parameters))
		return _ok(sched, 201)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/report/<processing_date>")
@_require_tenant
def get_eod_report(tenant_id: str, processing_date: str) -> Response:
	"""GET /api/fin/eod/report/2026-06-11"""
	try:
		report = _run(_SERVICE.get_eod_report(tenant_id, processing_date))
		return _ok(report)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/prerequisites/<eod_date>")
@_require_tenant
def check_eod_prerequisites(tenant_id: str, eod_date: str) -> Response:
	"""GET /api/fin/eod/prerequisites/2026-06-11"""
	result = _run(_SERVICE.check_eod_prerequisites(tenant_id, eod_date))
	return _ok(result)


@bp.get("/running")
@_require_tenant
def get_running_jobs(tenant_id: str) -> Response:
	"""GET /api/fin/eod/running"""
	jobs = _run(_SERVICE.get_running_jobs(tenant_id))
	return _ok(jobs)


@bp.post("/cancel/<eod_date>")
@_require_tenant
def cancel_running_eod(tenant_id: str, eod_date: str) -> Response:
	"""POST /api/fin/eod/cancel/2026-06-11
	Body: {"reason": "Emergency stop — incorrect rates loaded"}
	"""
	body   = request.get_json(silent=True) or {}
	reason = body.get("reason", "Operator cancel")
	result = _run(_SERVICE.cancel_running_eod(tenant_id, eod_date, reason))
	return _ok(result)


@bp.get("/metrics")
@_require_tenant
def get_eod_metrics(tenant_id: str) -> Response:
	"""GET /api/fin/eod/metrics?days=30"""
	days = int(request.args.get("days", 30))
	try:
		metrics = _run(_SERVICE.get_eod_metrics(tenant_id, days))
		return _ok(metrics)
	except Exception as ex:
		return _err(str(ex), 500)


@bp.get("/health")
def health_check() -> Response:
	"""GET /api/fin/eod/health — no auth required"""
	data = _run(_SERVICE.health_check())
	return _ok(data)


@bp.get("/job-types")
def list_job_types() -> Response:
	"""GET /api/fin/eod/job-types — enumerate available job names"""
	return _ok([{"name": j.value, "label": j.name} for j in EODJobType])


def get_blueprint() -> Blueprint:
	return bp
