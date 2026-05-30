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
