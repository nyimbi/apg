"""API helpers for the Workflow Orchestration capability."""

from __future__ import annotations

from typing import Any

from .service import WfloService


SERVICE = WfloService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"definition_count": summary["definition_count"],
		"execution_count": summary["execution_count"],
		"open_task_count": summary["open_task_count"],
		"pending_approval_count": summary["pending_approval_count"],
		"agent_count": summary["agent_count"],
		"pending_review_count": summary["pending_review_count"],
		"lifecycle_batch_count": summary["lifecycle_batch_count"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def create_workflow_definition(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_workflow_definition(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner_ref=str(payload.get("owner_ref") or ""),
		steps=list(payload.get("steps") or []),
		trigger_type=str(payload.get("trigger_type") or "manual"),
		trigger_policy_ref=str(payload.get("trigger_policy_ref") or ""),
		retry_policy_ref=str(payload.get("retry_policy_ref") or ""),
		compensation_ref=str(payload.get("compensation_ref") or ""),
		expected_runtime_minutes=int(payload.get("expected_runtime_minutes", 60)),
		runtime_review_recorded=_payload_bool(payload, "runtime_review_recorded"),
		version=int(payload.get("version", 1)),
		actor=str(payload.get("actor") or "system"),
	)


def publish_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_workflow(
		tenant_id=str(payload.get("tenant_id") or "default"),
		definition_id=str(payload["definition_id"]),
		approval_ref=str(payload.get("approval_ref") or ""),
		published_by=str(payload.get("published_by") or "system"),
	)


def start_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.start_execution(
		tenant_id=str(payload.get("tenant_id") or "default"),
		definition_id=str(payload["definition_id"]),
		correlation_id=str(payload.get("correlation_id") or ""),
		started_by=str(payload.get("started_by") or "system"),
		payload=dict(payload.get("payload") or {}),
		event_stream=str(payload.get("event_stream") or "bytewax"),
	)


def create_task(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_task(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		step_id=str(payload["step_id"]),
		title=str(payload.get("title") or ""),
		assignee_ref=str(payload.get("assignee_ref") or ""),
		due_at=payload.get("due_at"),
	)


def complete_task(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_task(
		tenant_id=str(payload.get("tenant_id") or "default"),
		task_id=str(payload["task_id"]),
		completed_by=str(payload.get("completed_by") or "system"),
	)


def claim_task(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.claim_task(
		tenant_id=str(payload.get("tenant_id") or "default"),
		task_id=str(payload["task_id"]),
		claimed_by=str(payload.get("claimed_by") or "system"),
	)


def escalate_task(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.escalate_task(
		tenant_id=str(payload.get("tenant_id") or "default"),
		task_id=str(payload["task_id"]),
		escalated_by=str(payload.get("escalated_by") or "system"),
		reason=str(payload.get("reason") or ""),
	)


def request_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_approval(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		subject_ref=str(payload.get("subject_ref") or ""),
		approver_ref=str(payload.get("approver_ref") or ""),
		reason=str(payload.get("reason") or ""),
	)


def record_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_approval(
		tenant_id=str(payload.get("tenant_id") or "default"),
		approval_id=str(payload["approval_id"]),
		decision=str(payload.get("decision") or ""),
		decision_by=str(payload.get("decision_by") or "system"),
		decision_evidence_ref=str(payload.get("decision_evidence_ref") or ""),
		delegated_to=str(payload.get("delegated_to") or ""),
	)


def complete_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_execution(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		actor=str(payload.get("actor") or "system"),
	)


def cancel_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.cancel_execution(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		actor=str(payload.get("actor") or "system"),
		reason=str(payload.get("reason") or ""),
	)


def fail_execution(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.fail_execution(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		actor=str(payload.get("actor") or "system"),
		reason=str(payload.get("reason") or ""),
		compensation_requested=_payload_bool(payload, "compensation_requested"),
	)


def run_compensation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.run_compensation(
		tenant_id=str(payload.get("tenant_id") or "default"),
		execution_id=str(payload["execution_id"]),
		actor=str(payload.get("actor") or "system"),
	)


def retire_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_workflow(
		tenant_id=str(payload.get("tenant_id") or "default"),
		definition_id=str(payload["definition_id"]),
		approval_ref=str(payload.get("approval_ref") or ""),
		retired_by=str(payload.get("retired_by") or "system"),
	)


def register_workflow_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_workflow_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or ""),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope_ref=str(payload.get("scope_ref") or ""),
		registered_by=str(payload.get("registered_by") or ""),
		contribution_disclosed=_payload_bool(payload, "contribution_disclosed"),
		owner_ref=str(payload.get("owner_ref") or ""),
		purpose=str(payload.get("purpose") or ""),
		human_approval_required=_payload_bool(payload, "human_approval_required"),
	)


def validate_batch_mutation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_batch_mutation(str(payload.get("event_stream") or ""))


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or ""),
		mutation_count=int(payload.get("mutation_count", 0)),
		operation=str(payload.get("operation") or "workflow_agent_batch"),
		batch_id=payload.get("batch_id"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_pending_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_pending_reviews(tenant_id)


def list_workflow_orchestration(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"definitions": SERVICE.list_definitions(tenant_id),
		"executions": SERVICE.list_executions(tenant_id),
		"tasks": SERVICE.list_tasks(tenant_id),
		"approvals": SERVICE.list_approvals(tenant_id),
		"agents": SERVICE.list_agents(tenant_id),
		"lifecycle_batches": SERVICE.list_lifecycle_batches(tenant_id),
		"pending_reviews": SERVICE.list_pending_reviews(tenant_id),
		"events": SERVICE.list_events(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def _payload_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
	value = payload.get(key, default)
	if isinstance(value, bool):
		return value
	if value is None:
		return False
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return bool(value)
