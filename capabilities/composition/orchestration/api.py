"""Dependency-light API helpers for APG workflow orchestration."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import WorkflowOrchestrationService
except ImportError:
	from capability_contract import get_capability_contract
	from service import WorkflowOrchestrationService


_SERVICE = WorkflowOrchestrationService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	"""Return service status and contract metadata for generated applications."""
	contract = get_capability_contract(tenant_id)
	return {
		"ok": True,
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"provides": contract["provides"],
		"requires": contract["requires"],
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"streaming": contract["streaming"],
		"summary": _SERVICE.dashboard_summary(tenant_id),
	}


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	"""Create a workflow definition using the package service."""
	return _SERVICE.create_record(payload)


def list_records(tenant_id: str = "default") -> list[dict[str, Any]]:
	"""List workflow definitions for a tenant."""
	return _SERVICE.list_records(tenant_id)


def define_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	"""Define and validate a workflow graph."""
	return _SERVICE.define_workflow(
		payload["workflow_id"],
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["owner"],
		payload.get("version", "1.0.0"),
		payload["tasks"],
		payload.get("start_event", "manual"),
		payload.get("terminal_state", "completed"),
		transactional=payload.get("transactional", False),
		compensation_steps=payload.get("compensation_steps"),
	)


def release_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	"""Release a validated workflow definition."""
	return _SERVICE.release_workflow(
		payload["release_id"],
		payload.get("tenant_id", "default"),
		payload["workflow_definition_id"],
		payload["validation_evidence"],
		payload["rollback_plan"],
		dry_run_passed=payload.get("dry_run_passed", False),
		approved_by=payload.get("approved_by"),
	)


def start_execution(payload: dict[str, Any]) -> dict[str, Any]:
	"""Start a workflow execution with an idempotency key."""
	return _SERVICE.start_execution(
		payload["execution_id"],
		payload.get("tenant_id", "default"),
		payload["workflow_definition_id"],
		payload["idempotency_key"],
		payload.get("inputs"),
		risk_level=payload.get("risk_level", "normal"),
		reviewed_by=payload.get("reviewed_by"),
	)


def complete_task(payload: dict[str, Any]) -> dict[str, Any]:
	"""Complete an active task and advance the execution graph."""
	return _SERVICE.complete_task(
		payload.get("tenant_id", "default"),
		payload["execution_record_id"],
		payload["task_id"],
		payload.get("result"),
	)


def register_workflow_agent(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register an AI agent runtime for orchestration review work."""
	return _SERVICE.register_workflow_agent(
		payload.get("tenant_id", "default"),
		payload["name"],
		payload["runtime"],
		payload["role"],
		payload.get("instructions", ""),
	)


def service() -> WorkflowOrchestrationService:
	"""Return the in-process service used by generated application adapters."""
	return _SERVICE
