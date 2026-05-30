"""UI view-model helpers for the APG Digital Twin Framework capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import DtwnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: DtwnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DtwnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"twins": service.list_twins(tenant_id),
		"models": service.list_models(tenant_id),
		"recent_telemetry": service.list_telemetry(tenant_id)[-10:],
		"review_queue": [prediction for prediction in service.list_predictions(tenant_id) if prediction["review_required"]],
		"twin_agents": service.list_twin_agents(tenant_id),
		"audit_timeline": service.list_audit_events(tenant_id),
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def topology_model(service: DtwnService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"twins": service.list_twins(tenant_id),
		"links": service.list_topology(tenant_id),
		"routes": capability_routes(tenant_id),
	}


def simulation_lab_model(service: DtwnService, tenant_id: str) -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"simulations": service.list_simulations(tenant_id),
		"predictions": service.list_predictions(tenant_id),
	}


def twin_agents_model(
	service: DtwnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DtwnService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_twin_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["twin_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["twin_agents"]["allowed_roles"],
		"actions": ["register_twin_agent"],
		"guardrails": [
			"twin_agent_requires_registration",
			"twin_agent_runtime_supported",
			"twin_agent_role_supported",
			"twin_agent_requires_scope",
			"twin_agent_requires_disclosure",
		],
		"theme_component": contract["theme"]["components"]["agent_panel"],
	}


def audit_trail_model(
	service: DtwnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DtwnService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"guardrails": ["dtwn_state_change_requires_reason", "dtwn_state_change_requires_audit", "cross_tenant_twin_access_denied"],
		"actions": ["change_twin_status", "validate_batch_twin_mutation"],
	}


def analytics_model(
	service: DtwnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or DtwnService()
	summary = service.dashboard_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"signals": {
			"telemetry_per_twin": _safe_ratio(summary["telemetry_sample_count"], summary["twin_count"]),
			"simulation_per_model": _safe_ratio(summary["simulation_count"], summary["model_count"]),
			"review_queue_ratio": _safe_ratio(summary["review_required_prediction_count"], max(len(service.list_predictions(tenant_id)), 1)),
		},
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def _safe_ratio(numerator: int, denominator: int) -> float:
	if denominator <= 0:
		return 0.0
	return round(numerator / denominator, 4)
