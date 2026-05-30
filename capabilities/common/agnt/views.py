"""UI metadata and view models for APG AI agent composition."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import AgntService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	contract = get_capability_contract(tenant_id)
	return {
		"summary": service.composition_summary(tenant_id),
		"agents": service.list_agents(tenant_id),
		"teams": service.list_teams(tenant_id),
		"runtimes": service.list_runtimes(tenant_id),
		"runtime_approvals": service.list_runtime_approvals(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"routes": capability_routes(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def team_builder_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	return {
		"agents": service.list_agents(tenant_id),
		"teams": service.list_teams(tenant_id),
		"handoff_edge_fields": ["source", "target", "trigger", "condition"],
		"execution_modes": ["sequential", "parallel"],
	}


def runtime_manager_model(service: AgntService | None = None) -> dict[str, object]:
	service = service or AgntService()
	return {
		"runtimes": service.list_runtimes(),
		"runtime_approvals": service.list_runtime_approvals(),
		"required_fields": ["name", "kind", "approved", "sandbox_policy"],
		"known_runtime_names": ["local", "codex", "claude_code", "opencode", "pi"],
	}


def runtime_approval_queue_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	return {
		"pending_requests": [
			request for request in service.list_runtime_approvals(tenant_id)
			if request["decision"] == "pending"
		],
		"decided_requests": [
			request for request in service.list_runtime_approvals(tenant_id)
			if request["decision"] != "pending"
		],
		"actions": ["approve", "reject"],
		"required_fields": ["reviewer", "decision", "notes"],
	}


def governance_evidence_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	return {
		"summary": service.composition_summary(tenant_id),
		"runtime_approvals": service.list_runtime_approvals(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def audit_trail_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	return {
		"audit_events": service.list_audit_events(tenant_id),
		"event_types": [
			"runtime_approval_requested",
			"runtime_approval_decided",
			"runtime_registered",
			"agent_registered",
			"team_registered",
			"execution_plan_built",
		],
	}


def analytics_model(
	service: AgntService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AgntService()
	summary = service.composition_summary(tenant_id)
	return {
		"summary": summary,
		"agents_per_team": _safe_ratio(summary["agent_count"], summary["team_count"]),
		"approval_density": _safe_ratio(summary["runtime_approval_count"], summary["runtime_count"]),
		"audit_events_per_agent": _safe_ratio(summary["audit_event_count"], summary["agent_count"]),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def execution_trace_model(plan: dict[str, object]) -> dict[str, object]:
	return {
		"plan_id": plan["id"],
		"team_id": plan["team_id"],
		"steps": plan["steps"],
		"approvals_required": plan["approvals_required"],
		"trace_columns": ["order", "agent", "runtime", "model", "handoff_targets"],
	}


def _safe_ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
