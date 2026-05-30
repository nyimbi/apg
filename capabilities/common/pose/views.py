"""UI metadata helpers for the APG Pose Estimation capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import PoseService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"models": service.list_models(tenant_id),
		"sessions": service.list_sessions(tenant_id),
		"estimates": service.list_estimates(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def estimator_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("estimate", tenant_id),
		"models": service.list_models(tenant_id),
		"frames": service.list_frames(tenant_id),
		"estimates": service.list_estimates(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["pose_viewer"],
	}


def tracking_console_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("tracking", tenant_id),
		"sessions": service.list_sessions(tenant_id),
		"frames": service.list_frames(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["tracking_timeline"],
	}


def analysis_workbench_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("analysis", tenant_id),
		"estimates": service.list_estimates(tenant_id),
		"analyses": service.list_analyses(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["biomechanics_panel"],
	}


def reconstruction_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("reconstruction", tenant_id),
		"reconstructions": service.list_reconstructions(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["reconstruction_panel"],
	}


def session_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("sessions", tenant_id),
		"sessions": service.list_sessions(tenant_id),
		"frames": service.list_frames(tenant_id),
	}


def model_registry_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("models", tenant_id),
		"models": service.list_models(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["model_registry"],
	}


def quality_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("quality", tenant_id),
		"estimates": service.list_estimates(tenant_id),
		"analyses": service.list_analyses(tenant_id),
	}


def pose_agents_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("agents", tenant_id),
		"agents": service.list_agents(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["agent_panel"],
	}


def audit_trail_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("audit", tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def analytics_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	return {
		"tenant_id": tenant_id,
		"route": _route("analytics", tenant_id),
		"summary": service.dashboard_summary(tenant_id),
	}


def settings_model(service: PoseService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or PoseService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": _route("settings", tenant_id),
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"permissions": sorted({route["permission"] for route in contract["ui"]["routes"]}),
	}


def _route(name: str, tenant_id: str) -> dict[str, str]:
	for route in capability_routes(tenant_id):
		if route["name"] == name:
			return route
	raise KeyError(f"pose_route_not_found:{name}")
