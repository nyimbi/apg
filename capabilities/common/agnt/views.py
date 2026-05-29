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
	return {
		"summary": service.composition_summary(tenant_id),
		"agents": service.list_agents(tenant_id),
		"teams": service.list_teams(tenant_id),
		"runtimes": service.list_runtimes(),
		"routes": capability_routes(tenant_id),
		"theme": get_capability_contract(tenant_id)["theme"],
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
		"required_fields": ["name", "kind", "approved", "sandbox_policy"],
		"known_runtime_names": ["local", "codex", "claude_code", "opencode", "pi"],
	}


def execution_trace_model(plan: dict[str, object]) -> dict[str, object]:
	return {
		"plan_id": plan["id"],
		"team_id": plan["team_id"],
		"steps": plan["steps"],
		"approvals_required": plan["approvals_required"],
		"trace_columns": ["order", "agent", "runtime", "model", "handoff_targets"],
	}
