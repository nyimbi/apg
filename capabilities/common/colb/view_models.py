"""Dependency-light view models for COLB generated applications."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .collaboration_runtime import CollaborationRuntime


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	contract = runtime.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": runtime.dashboard_summary(tenant_id),
		"workspaces": runtime.list_workspaces(tenant_id),
		"sessions": runtime.list_sessions(tenant_id),
		"artifacts": runtime.list_artifacts(tenant_id),
		"annotations": runtime.list_annotations(tenant_id),
		"decisions": runtime.list_decisions(tenant_id),
		"presence": runtime.list_presence(tenant_id),
		"collaboration_agents": runtime.list_collaboration_agents(tenant_id),
		"lifecycle_batches": runtime.list_lifecycle_batches(tenant_id),
		"audit_events": runtime.list_audit_events(tenant_id),
		"theme": contract["theme"],
	}


def workspace_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	workspaces = runtime.list_workspaces(tenant_id)
	return {
		"tenant_id": tenant_id,
		"workspaces": workspaces,
		"active": [item for item in workspaces if item["status"] == "active"],
		"pending_review": [item for item in workspaces if item["status"] == "pending_review"],
	}


def session_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	return {"tenant_id": tenant_id, "sessions": runtime.list_sessions(tenant_id), "presence": runtime.list_presence(tenant_id)}


def artifact_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	return {
		"tenant_id": tenant_id,
		"artifacts": runtime.list_artifacts(tenant_id),
		"annotations": runtime.list_annotations(tenant_id),
		"decisions": runtime.list_decisions(tenant_id),
	}


def agent_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"enabled": contract["configuration"]["ai_agents"]["agent_collaboration_enabled"],
		"supported_runtimes": contract["configuration"]["ai_agents"]["supported_runtimes"],
		"required_controls": [
			"agent_registration_required",
			"agent_scope_required",
			"agent_contribution_disclosure_required",
		],
		"theme": contract["theme"]["components"]["agent_panel"],
	}


def collaboration_agent_roster_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	contract = runtime.describe(tenant_id)
	agents = runtime.list_collaboration_agents(tenant_id)
	return {
		"route": "/colb/agents",
		"tenant_id": tenant_id,
		"agents": agents,
		"active": [agent for agent in agents if agent["status"] == "active"],
		"pending_review": [agent for agent in agents if agent["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"actions": ["register_collaboration_agent", "record_human_collaboration_agent_approval"],
		"theme_component": "collaboration_agent_roster",
	}


def lifecycle_batch_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	contract = runtime.describe(tenant_id)
	batches = runtime.list_lifecycle_batches(tenant_id)
	return {
		"route": "/colb/lifecycle",
		"tenant_id": tenant_id,
		"lifecycle_stream": contract["streaming"]["lifecycle_stream"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"batches": batches,
		"accepted": [batch for batch in batches if batch["status"] == "accepted"],
		"denied": [batch for batch in batches if batch["status"] == "denied"],
		"actions": ["validate_lifecycle_batch", "inspect_bytewax_lifecycle"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def analytics_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	summary = runtime.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"artifact_density": summary["artifact_count"] / summary["workspace_count"] if summary["workspace_count"] else 0.0,
		"decision_density": summary["decision_count"] / summary["annotation_count"] if summary["annotation_count"] else 0.0,
	}


def audit_model(runtime: CollaborationRuntime | None = None, tenant_id: str = "default") -> dict[str, object]:
	runtime = runtime or CollaborationRuntime()
	return {"tenant_id": tenant_id, "audit_events": runtime.list_audit_events(tenant_id)}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "configuration": contract["configuration"], "rules": contract["rule_engine"]["rules"], "theme": contract["theme"]}
