"""Temporal workflow capability — REST API endpoints."""
from __future__ import annotations

import logging
from typing import Any

from flask import Blueprint, jsonify, request

from .service import TemporalService
from .models import (
	StartWorkflowRequest,
	SignalWorkflowRequest,
	CompleteTaskRequest,
	FailTaskRequest,
	ScheduleWorkflowRequest,
)

_log = logging.getLogger(__name__)

temporal_api = Blueprint("temporal_api", __name__, url_prefix="/api/temporal")


def _svc() -> TemporalService:
	return TemporalService(tenant_id=request.headers.get("X-Tenant-Id", "default"))


# ── Health ────────────────────────────────────────────────────────────────────

@temporal_api.get("/health")
async def health():
	svc = _svc()
	await svc.connect()
	result = await svc.health_check()
	await svc.disconnect()
	return jsonify(result)


@temporal_api.get("/system")
async def system_info():
	svc = _svc()
	await svc.connect()
	result = await svc.get_system_info()
	await svc.disconnect()
	return jsonify(result)


# ── Workflows ─────────────────────────────────────────────────────────────────

@temporal_api.post("/workflows")
async def start_workflow():
	body = StartWorkflowRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	try:
		await svc.connect()
		result = await svc.start_workflow(
			body.workflow_type,
			body.workflow_id,
			input_data=body.input_data,
			task_queue=body.task_queue,
			execution_timeout_seconds=body.execution_timeout_seconds,
		)
		await svc.disconnect()
	except Exception as exc:
		_log.exception("start_workflow failed")
		return jsonify({"error": str(exc)}), 500
	return jsonify(result), 201


@temporal_api.get("/workflows")
async def list_workflows():
	status = request.args.get("status", "")
	workflow_type = request.args.get("workflow_type", "")
	limit = int(request.args.get("limit", "50"))
	svc = _svc()
	await svc.connect()
	workflows = await svc.list_workflows(status=status, workflow_type=workflow_type, limit=limit)
	await svc.disconnect()
	return jsonify({"workflows": workflows, "total": len(workflows)})


@temporal_api.get("/workflows/<workflow_id>")
async def get_workflow(workflow_id: str):
	svc = _svc()
	await svc.connect()
	result = await svc.get_workflow_status(workflow_id)
	await svc.disconnect()
	if result.get("status") == "NOT_FOUND":
		return jsonify(result), 404
	return jsonify(result)


@temporal_api.delete("/workflows/<workflow_id>")
async def cancel_workflow(workflow_id: str):
	reason = request.args.get("reason", "")
	svc = _svc()
	await svc.connect()
	result = await svc.cancel_workflow(workflow_id, reason=reason)
	await svc.disconnect()
	return jsonify(result)


@temporal_api.post("/workflows/<workflow_id>/signal")
async def signal_workflow(workflow_id: str):
	body = SignalWorkflowRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	await svc.connect()
	result = await svc.signal_workflow(workflow_id, body.signal_name, payload=body.payload)
	await svc.disconnect()
	return jsonify(result)


@temporal_api.get("/workflows/<workflow_id>/history")
async def get_workflow_history(workflow_id: str):
	svc = _svc()
	await svc.connect()
	history = await svc.get_workflow_history(workflow_id)
	await svc.disconnect()
	return jsonify({"workflow_id": workflow_id, "history": history})


# ── Human tasks ───────────────────────────────────────────────────────────────

@temporal_api.post("/tasks/complete")
async def complete_task():
	body = CompleteTaskRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	await svc.connect()
	result = await svc.complete_task(body.task_token, result=body.result)
	await svc.disconnect()
	return jsonify(result)


@temporal_api.post("/tasks/fail")
async def fail_task():
	body = FailTaskRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	await svc.connect()
	result = await svc.fail_task(body.task_token, error=body.error)
	await svc.disconnect()
	return jsonify(result)


# ── Schedules ─────────────────────────────────────────────────────────────────

@temporal_api.post("/schedules")
async def create_schedule():
	body = ScheduleWorkflowRequest.model_validate(request.get_json(force=True))
	svc = _svc()
	await svc.connect()
	result = await svc.schedule_workflow(
		body.schedule_id,
		body.workflow_type,
		body.cron_expression,
		input_data=body.input_data,
	)
	await svc.disconnect()
	return jsonify(result), 201


@temporal_api.get("/schedules")
async def list_schedules():
	svc = _svc()
	await svc.connect()
	schedules = await svc.list_schedules()
	await svc.disconnect()
	return jsonify({"schedules": schedules, "total": len(schedules)})


@temporal_api.delete("/schedules/<schedule_id>")
async def delete_schedule(schedule_id: str):
	svc = _svc()
	await svc.connect()
	result = await svc.delete_schedule(schedule_id)
	await svc.disconnect()
	return jsonify(result)


@temporal_api.post("/schedules/<schedule_id>/pause")
async def pause_schedule(schedule_id: str):
	note = (request.get_json(force=True) or {}).get("note", "")
	svc = _svc()
	await svc.connect()
	result = await svc.pause_schedule(schedule_id, note=note)
	await svc.disconnect()
	return jsonify(result)


@temporal_api.post("/schedules/<schedule_id>/resume")
async def resume_schedule(schedule_id: str):
	note = (request.get_json(force=True) or {}).get("note", "")
	svc = _svc()
	await svc.connect()
	result = await svc.resume_schedule(schedule_id, note=note)
	await svc.disconnect()
	return jsonify(result)


# ── Metrics ───────────────────────────────────────────────────────────────────

@temporal_api.get("/metrics")
async def get_metrics():
	svc = _svc()
	await svc.connect()
	metrics = await svc.get_metrics()
	await svc.disconnect()
	return jsonify(metrics)


@temporal_api.get("/namespaces")
async def list_namespaces():
	svc = _svc()
	await svc.connect()
	namespaces = await svc.list_namespaces()
	await svc.disconnect()
	return jsonify({"namespaces": namespaces})
