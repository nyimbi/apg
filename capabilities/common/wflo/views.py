"""UI metadata helpers for the Workflow Orchestration capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import WfloService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def designer_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/designer",
		"tenant_id": tenant_id,
		"definitions": service.list_definitions(tenant_id),
		"step_types": contract_step_types(),
		"versioning_enabled": True,
		"required_policies": ["retry_policy_ref", "trigger_policy_ref", "ai_policy_ref", "automation_policy_ref", "event_policy_ref"],
	}


def definition_library_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/definitions",
		"tenant_id": tenant_id,
		"definitions": service.list_definitions(tenant_id),
		"pending_reviews": [
			definition
			for definition in service.list_pending_reviews(tenant_id)
			if definition.get("trigger_type")
		],
		"statuses": ["draft", "review_required", "published", "retired"],
	}


def execution_monitor_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/executions",
		"tenant_id": tenant_id,
		"executions": service.list_executions(tenant_id),
		"events": service.list_events(tenant_id),
		"statuses": ["running", "waiting_approval", "completed", "failed", "cancelled"],
		"compensation_states": ["not_required", "available", "requested", "completed"],
	}


def task_inbox_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/tasks",
		"tenant_id": tenant_id,
		"tasks": service.list_tasks(tenant_id),
		"statuses": ["open", "claimed", "completed", "escalated"],
		"required_controls": ["assignee_ref", "claimed_by", "escalation_reason"],
	}


def approval_center_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/approvals",
		"tenant_id": tenant_id,
		"approvals": service.list_approvals(tenant_id),
		"statuses": ["pending", "approved", "rejected", "delegated"],
		"required_controls": ["approver_ref", "reason", "decision_evidence_ref", "delegated_to"],
	}


def agent_panel_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	contract = service.describe(tenant_id)
	return {
		"route": "/wflo/agents",
		"tenant_id": tenant_id,
		"agents": service.list_agents(tenant_id),
		"pending_reviews": [
			agent
			for agent in service.list_agents(tenant_id)
			if agent["status"] == "pending_review"
		],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"required_controls": ["registered_by", "owner_ref", "purpose", "scope_ref", "contribution_disclosed", "human_approval_required"],
		"theme_component": contract["theme"]["components"]["workflow_agent_roster"],
	}


def lifecycle_batch_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	contract = service.describe(tenant_id)
	return {
		"route": "/wflo/lifecycle",
		"tenant_id": tenant_id,
		"batches": service.list_lifecycle_batches(tenant_id),
		"denied": [
			batch
			for batch in service.list_lifecycle_batches(tenant_id)
			if batch["status"] == "denied"
		],
		"streaming": contract["streaming"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"theme_component": contract["theme"]["components"]["bytewax_lifecycle_panel"],
	}


def audit_trail_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/audit",
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"event_types": sorted({event["event_type"] for event in service.list_audit_events(tenant_id)}),
	}


def analytics_model(
	service: WfloService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WfloService()
	return {
		"route": "/wflo/analytics",
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"review_required_definitions": [
			definition
			for definition in service.list_definitions(tenant_id)
			if definition["status"] == "review_required"
		],
		"execution_states": {state: len([item for item in service.list_executions(tenant_id) if item["status"] == state]) for state in ["running", "waiting_approval", "completed", "failed", "cancelled"]},
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/wflo/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def contract_step_types(tenant_id: str = "default") -> list[str]:
	return list(get_capability_contract(tenant_id)["configuration"]["steps"]["supported_step_types"])
