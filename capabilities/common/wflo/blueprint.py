"""Flask Blueprint REST API for the Workflow Orchestration capability.

URL prefix: /api/wflo

All endpoints enforce tenant isolation via X-Tenant-ID header (falls back to
query param or body field).  Authentication is delegated to the auth adapter
(NullAuthAdapter in standalone mode).

Endpoint map
────────────
Definitions
  GET    /api/wflo/definitions              list with filter/sort/page
  POST   /api/wflo/definitions              create
  GET    /api/wflo/definitions/<id>         detail
  PUT    /api/wflo/definitions/<id>         update
  DELETE /api/wflo/definitions/<id>         soft-delete
  POST   /api/wflo/definitions/<id>/publish
  POST   /api/wflo/definitions/<id>/retire

Instances
  GET    /api/wflo/instances               list
  POST   /api/wflo/instances               start instance
  GET    /api/wflo/instances/<id>          detail
  POST   /api/wflo/instances/<id>/suspend
  POST   /api/wflo/instances/<id>/resume
  POST   /api/wflo/instances/<id>/cancel
  POST   /api/wflo/instances/<id>/complete
  POST   /api/wflo/instances/<id>/migrate

Tasks
  GET    /api/wflo/tasks                   list (my tasks, overdue, all)
  GET    /api/wflo/tasks/<id>              detail
  PUT    /api/wflo/tasks/<id>              update assignment / priority
  POST   /api/wflo/tasks/<id>/claim
  POST   /api/wflo/tasks/<id>/complete
  POST   /api/wflo/tasks/<id>/escalate

Timers
  GET    /api/wflo/timers                  list pending
  POST   /api/wflo/timers/process          trigger timer processing

Gateways
  POST   /api/wflo/gateways/<id>/evaluate  evaluate gateway conditions

Escalations
  GET    /api/wflo/escalations             list active
  POST   /api/wflo/escalations/<id>/resolve

Compensations
  GET    /api/wflo/compensations           list
  POST   /api/wflo/compensations/trigger

Variables
  GET    /api/wflo/instances/<id>/variables
  POST   /api/wflo/instances/<id>/variables
  PUT    /api/wflo/instances/<id>/variables/<name>

History
  GET    /api/wflo/instances/<id>/history

Reports
  GET    /api/wflo/reports/analytics       workflow analytics
  GET    /api/wflo/reports/sla             SLA snapshot
  GET    /api/wflo/reports/dashboard       KPI dashboard
"""
from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any

from flask import Blueprint, jsonify, request

from .async_service import WorkflowOrchestrationService
from .domain.adapters import get_auth_adapter, get_audit_adapter, get_notify_adapter
from .domain.rules import RuleViolation
from .models import (
	WorkflowDefinitionCreate,
	WorkflowDefinitionUpdate,
	WorkflowInstanceCreate,
	TaskCreate,
	TaskUpdate,
	WorkflowVariableCreate,
)

log = logging.getLogger(__name__)

wflo_bp = Blueprint("wflo", __name__, url_prefix="/api/wflo")

# Singleton service — swapped in tests
_service: WorkflowOrchestrationService | None = None


def _get_service() -> WorkflowOrchestrationService:
	global _service
	if _service is None:
		_service = WorkflowOrchestrationService(
			auth=get_auth_adapter(),
			audit=get_audit_adapter(),
			notify=get_notify_adapter(),
		)
	return _service


def _tenant() -> str:
	"""Extract tenant_id from header → query → body (in that order)."""
	tid = (
		request.headers.get("X-Tenant-ID")
		or request.args.get("tenant_id")
		or (request.json or {}).get("tenant_id")
		or "default"
	)
	return str(tid).strip() or "default"


def _actor() -> str:
	auth_header = request.headers.get("Authorization", "")
	if auth_header.startswith("Bearer "):
		return auth_header[7:].strip() or "anonymous"
	return request.headers.get("X-Actor-ID", "anonymous")


def _run(coro: Any) -> Any:
	"""Run an async coroutine from a sync Flask route."""
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	return loop.run_until_complete(coro)


def _ok(data: Any, status: int = 200):
	return jsonify({"ok": True, "data": data}), status


def _err(message: str, status: int = 400, detail: dict[str, Any] | None = None):
	body: dict[str, Any] = {"ok": False, "error": message}
	if detail:
		body["detail"] = detail
	return jsonify(body), status


def _handle(fn):
	"""Decorator: catch RuleViolation and generic exceptions."""
	@wraps(fn)
	def wrapper(*args, **kwargs):
		try:
			return fn(*args, **kwargs)
		except RuleViolation as exc:
			return _err(exc.reason, 422, {
				"rule": exc.rule_name,
				"required_action": exc.required_action,
				**exc.details,
			})
		except (KeyError, LookupError) as exc:
			return _err(str(exc), 404)
		except (ValueError, TypeError) as exc:
			return _err(str(exc), 400)
		except PermissionError as exc:
			return _err(str(exc), 403)
		except Exception as exc:
			log.exception("Unhandled error in wflo blueprint")
			return _err("internal_error", 500)
	return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Definitions
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/definitions")
@_handle
def list_definitions():
	"""List workflow definitions with optional filtering and pagination."""
	svc = _get_service()
	tenant = _tenant()
	params = {
		"status": request.args.get("status"),
		"category": request.args.get("category"),
		"search": request.args.get("q"),
		"page": int(request.args.get("page", 1)),
		"page_size": int(request.args.get("page_size", 20)),
		"sort_by": request.args.get("sort_by", "created_at"),
		"sort_dir": request.args.get("sort_dir", "desc"),
	}
	result = _run(svc.list_definitions(tenant, **params))
	return _ok(result)


@wflo_bp.post("/definitions")
@_handle
def create_definition():
	"""Create a new workflow definition."""
	svc = _get_service()
	body = request.json or {}
	body["tenant_id"] = _tenant()
	body["created_by"] = _actor()
	payload = WorkflowDefinitionCreate(**body)
	result = _run(svc.create_definition(payload))
	return _ok(result, 201)


@wflo_bp.get("/definitions/<definition_id>")
@_handle
def get_definition(definition_id: str):
	"""Get a single workflow definition with computed stats."""
	svc = _get_service()
	result = _run(svc.get_definition(definition_id, _tenant()))
	return _ok(result)


@wflo_bp.put("/definitions/<definition_id>")
@_handle
def update_definition(definition_id: str):
	"""Update a draft workflow definition."""
	svc = _get_service()
	body = request.json or {}
	payload = WorkflowDefinitionUpdate(**body)
	result = _run(svc.update_definition(definition_id, payload, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.delete("/definitions/<definition_id>")
@_handle
def delete_definition(definition_id: str):
	"""Soft-delete a workflow definition."""
	svc = _get_service()
	result = _run(svc.delete_definition(definition_id, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/definitions/<definition_id>/publish")
@_handle
def publish_definition(definition_id: str):
	"""Publish a workflow definition (requires approval_ref)."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.publish_definition(
		definition_id, _tenant(), _actor(),
		approval_ref=str(body.get("approval_ref", "")),
	))
	return _ok(result)


@wflo_bp.post("/definitions/<definition_id>/retire")
@_handle
def retire_definition(definition_id: str):
	"""Retire a workflow definition."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.retire_definition(
		definition_id, _tenant(), _actor(),
		approval_ref=str(body.get("approval_ref", "")),
	))
	return _ok(result)


@wflo_bp.post("/definitions/<definition_id>/deploy")
@_handle
def deploy_definition(definition_id: str):
	"""Deploy a BPMN/APG workflow definition (parse + validate + persist)."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.deploy_workflow(
		bpmn_or_apg=body.get("source", ""),
		tenant_id=_tenant(),
		actor_id=_actor(),
		definition_id=definition_id,
	))
	return _ok(result, 201)


# ─────────────────────────────────────────────────────────────────────────────
# Instances
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/instances")
@_handle
def list_instances():
	"""List workflow instances with filtering and pagination."""
	svc = _get_service()
	tenant = _tenant()
	params = {
		"definition_id": request.args.get("definition_id"),
		"status": request.args.get("status"),
		"sla_breached": request.args.get("sla_breached"),
		"page": int(request.args.get("page", 1)),
		"page_size": int(request.args.get("page_size", 20)),
		"sort_by": request.args.get("sort_by", "created_at"),
		"sort_dir": request.args.get("sort_dir", "desc"),
	}
	result = _run(svc.list_instances(tenant, **params))
	return _ok(result)


@wflo_bp.post("/instances")
@_handle
def start_instance():
	"""Start a new workflow instance from a published definition."""
	svc = _get_service()
	body = request.json or {}
	body["tenant_id"] = _tenant()
	body["created_by"] = _actor()
	payload = WorkflowInstanceCreate(**body)
	result = _run(svc.start_instance(payload))
	return _ok(result, 201)


@wflo_bp.get("/instances/<instance_id>")
@_handle
def get_instance(instance_id: str):
	"""Get a workflow instance with open tasks and pending approvals count."""
	svc = _get_service()
	result = _run(svc.get_instance(instance_id, _tenant()))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/suspend")
@_handle
def suspend_instance(instance_id: str):
	"""Suspend a running workflow instance."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.suspend_instance(
		instance_id, _tenant(), _actor(),
		reason=str(body.get("reason", "")),
	))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/resume")
@_handle
def resume_instance(instance_id: str):
	"""Resume a suspended workflow instance."""
	svc = _get_service()
	result = _run(svc.resume_instance(instance_id, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/cancel")
@_handle
def cancel_instance(instance_id: str):
	"""Cancel an active workflow instance."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.cancel_instance(
		instance_id, _tenant(), _actor(),
		reason=str(body.get("reason", "")),
	))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/complete")
@_handle
def complete_instance(instance_id: str):
	"""Mark a workflow instance as completed (all tasks must be done)."""
	svc = _get_service()
	result = _run(svc.complete_instance(instance_id, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/migrate")
@_handle
def migrate_instance(instance_id: str):
	"""Migrate an active instance to a newer definition version."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.migrate_instance(
		instance_id, _tenant(), _actor(),
		new_version=int(body.get("new_version", 0)),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Tasks
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/tasks")
@_handle
def list_tasks():
	"""List tasks. Filter by assignee, status, overdue, instance."""
	svc = _get_service()
	tenant = _tenant()
	params = {
		"assignee_ref": request.args.get("assignee"),
		"instance_id": request.args.get("instance_id"),
		"status": request.args.get("status"),
		"overdue": request.args.get("overdue") == "true",
		"page": int(request.args.get("page", 1)),
		"page_size": int(request.args.get("page_size", 20)),
	}
	result = _run(svc.list_tasks(tenant, **params))
	return _ok(result)


@wflo_bp.get("/tasks/<task_id>")
@_handle
def get_task(task_id: str):
	"""Get a single task with overdue flag and minutes_until_due."""
	svc = _get_service()
	result = _run(svc.get_task(task_id, _tenant()))
	return _ok(result)


@wflo_bp.put("/tasks/<task_id>")
@_handle
def update_task(task_id: str):
	"""Update task assignment, priority, or description."""
	svc = _get_service()
	body = request.json or {}
	payload = TaskUpdate(**body)
	result = _run(svc.update_task(task_id, payload, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/tasks/<task_id>/claim")
@_handle
def claim_task(task_id: str):
	"""Claim a task — sets claimed_by to the acting user."""
	svc = _get_service()
	result = _run(svc.claim_task(task_id, _tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/tasks/<task_id>/complete")
@_handle
def complete_task(task_id: str):
	"""Complete a claimed task with outcome and output variables."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.complete_task(
		task_id=task_id,
		tenant_id=_tenant(),
		actor_id=_actor(),
		outcome=str(body.get("outcome", "")),
		output_variables=dict(body.get("variables", {})),
	))
	return _ok(result)


@wflo_bp.post("/tasks/<task_id>/escalate")
@_handle
def escalate_task(task_id: str):
	"""Escalate a task to a higher authority."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.escalate_task(
		task_id=task_id,
		tenant_id=_tenant(),
		actor_id=_actor(),
		reason=str(body.get("reason", "")),
		escalated_to=str(body.get("escalated_to", "")),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Timers
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/timers")
@_handle
def list_timers():
	"""List pending (unfired) timers for the tenant."""
	svc = _get_service()
	result = _run(svc.list_timers(_tenant()))
	return _ok(result)


@wflo_bp.post("/timers/process")
@_handle
def process_timers():
	"""Trigger timer event processing — fires all due timers."""
	svc = _get_service()
	result = _run(svc.timer_event_processing(_tenant()))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Gateways
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.post("/gateways/<gateway_id>/evaluate")
@_handle
def evaluate_gateway(gateway_id: str):
	"""Evaluate a gateway's conditions against current instance variables."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.evaluate_gateway(
		gateway_id=gateway_id,
		tenant_id=_tenant(),
		actor_id=_actor(),
		variables=dict(body.get("variables", {})),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Boundary events
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.post("/boundary-events/<event_id>/trigger")
@_handle
def trigger_boundary_event(event_id: str):
	"""Trigger a boundary event attached to a task."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.process_boundary_event(
		event_id=event_id,
		tenant_id=_tenant(),
		actor_id=_actor(),
		trigger_data=dict(body.get("data", {})),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Escalations
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/escalations")
@_handle
def list_escalations():
	"""List active escalations for the tenant."""
	svc = _get_service()
	instance_id = request.args.get("instance_id")
	result = _run(svc.list_escalations(_tenant(), instance_id=instance_id))
	return _ok(result)


@wflo_bp.post("/escalations/<escalation_id>/resolve")
@_handle
def resolve_escalation(escalation_id: str):
	"""Resolve an active escalation."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.resolve_escalation(
		escalation_id=escalation_id,
		tenant_id=_tenant(),
		actor_id=_actor(),
		note=str(body.get("note", "")),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Compensations
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/compensations")
@_handle
def list_compensations():
	"""List compensation activities for the tenant."""
	svc = _get_service()
	result = _run(svc.list_compensations(_tenant()))
	return _ok(result)


@wflo_bp.post("/compensations/trigger")
@_handle
def trigger_compensation():
	"""Trigger compensation for a failed instance."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.compensation_handling(
		instance_id=str(body.get("instance_id", "")),
		tenant_id=_tenant(),
		actor_id=_actor(),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Variables
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/instances/<instance_id>/variables")
@_handle
def list_variables(instance_id: str):
	"""List all variables for a workflow instance."""
	svc = _get_service()
	result = _run(svc.list_variables(instance_id, _tenant()))
	return _ok(result)


@wflo_bp.post("/instances/<instance_id>/variables")
@_handle
def set_variable(instance_id: str):
	"""Create or update a workflow variable."""
	svc = _get_service()
	body = request.json or {}
	body["instance_id"] = instance_id
	body["tenant_id"] = _tenant()
	body["created_by"] = _actor()
	payload = WorkflowVariableCreate(**body)
	result = _run(svc.set_variable(payload))
	return _ok(result, 201)


@wflo_bp.put("/instances/<instance_id>/variables/<name>")
@_handle
def update_variable(instance_id: str, name: str):
	"""Update the value of an existing workflow variable."""
	svc = _get_service()
	body = request.json or {}
	result = _run(svc.update_variable(
		instance_id=instance_id,
		name=name,
		value=body.get("value"),
		tenant_id=_tenant(),
		actor_id=_actor(),
	))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/instances/<instance_id>/history")
@_handle
def get_history(instance_id: str):
	"""Get the full event history for a workflow instance."""
	svc = _get_service()
	result = _run(svc.get_instance_history(instance_id, _tenant()))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Operational actions
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.post("/ops/escalate-overdue")
@_handle
def escalate_overdue():
	"""Scan for overdue tasks and escalate them automatically."""
	svc = _get_service()
	result = _run(svc.escalate_overdue_tasks(_tenant(), _actor()))
	return _ok(result)


@wflo_bp.post("/ops/sla-check")
@_handle
def sla_check():
	"""Run SLA monitoring scan and return breached/at-risk instances."""
	svc = _get_service()
	result = _run(svc.sla_monitoring(_tenant()))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/reports/analytics")
@_handle
def report_analytics():
	"""Workflow analytics for a definition over a time window."""
	svc = _get_service()
	params = {
		"definition_id": request.args.get("definition_id"),
		"period_days": int(request.args.get("period_days", 30)),
	}
	result = _run(svc.workflow_analytics(_tenant(), **params))
	return _ok(result)


@wflo_bp.get("/reports/sla")
@_handle
def report_sla():
	"""SLA monitoring snapshot — active instances categorised by health."""
	svc = _get_service()
	result = _run(svc.sla_monitoring(_tenant()))
	return _ok(result)


@wflo_bp.get("/reports/dashboard")
@_handle
def report_dashboard():
	"""Real-time KPI dashboard for the tenant."""
	svc = _get_service()
	result = _run(svc.dashboard_kpis(_tenant(), _actor()))
	return _ok(result)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────

@wflo_bp.get("/health")
def health():
	"""Capability health check."""
	return _ok({"capability": "wflo", "status": "healthy"})
