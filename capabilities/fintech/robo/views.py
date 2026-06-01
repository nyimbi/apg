"""View models for APG Robo Advisory."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RoboAdvisoryService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RoboAdvisoryService  # type: ignore


def dashboard_model(service: RoboAdvisoryService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Robo Advisory",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Profiles", "value": summary["profile_count"], "icon": "user-round-check"},
			{"label": "Goals", "value": summary["goal_count"], "icon": "target"},
			{"label": "Models", "value": summary["model_count"], "icon": "pie-chart"},
			{"label": "Recommendations", "value": summary["recommendation_count"], "icon": "sparkles"},
			{"label": "Automation", "value": summary["automation_count"], "icon": "repeat"},
			{"label": "Agents", "value": len([item for item in service.evidence.values() if item.tenant_id == tenant_id and item.kind == "agent"]), "icon": "bot"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def robo_console_model(service: RoboAdvisoryService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"profiles": _items(service.profiles, tenant_id),
		"goals": _items(service.goals, tenant_id),
		"models": _items(service.models, tenant_id),
		"recommendations": _items(service.recommendations, tenant_id),
		"automation": _items(service.automation, tenant_id),
		"drift": _items(service.drift, tenant_id),
		"tax_loss": _items(service.tax_loss, tenant_id),
		"reviews": _items(service.reviews, tenant_id),
		"agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "agent"],
	}


def route_models(service: RoboAdvisoryService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = robo_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}


def _items(items: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(items.values(), key=lambda item: item.id) if item.tenant_id == tenant_id]
