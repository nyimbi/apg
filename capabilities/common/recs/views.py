"""UI metadata helpers for the Recommender Systems capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import RecsService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"models": service.list_models(tenant_id),
		"recommendation_sets": service.list_recommendation_sets(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def recommendation_console_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("recommendations", tenant_id),
		"recommendation_sets": service.list_recommendation_sets(tenant_id),
		"profiles": service.list_profiles(tenant_id),
		"policies": service.list_policies(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["recommendation_list"],
	}


def model_registry_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("models", tenant_id),
		"models": service.list_models(tenant_id),
		"training_runs": service.list_training_runs(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["model_card"],
	}


def catalog_manager_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("catalogs", tenant_id),
		"catalog_items": service.list_catalog_items(tenant_id),
	}


def profile_features_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("profiles", tenant_id),
		"profiles": service.list_profiles(tenant_id),
	}


def experiment_studio_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("experiments", tenant_id),
		"experiments": service.list_experiments(tenant_id),
		"models": service.list_models(tenant_id),
		"policies": service.list_policies(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["experiment_board"],
	}


def ranking_policy_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	return {
		"tenant_id": tenant_id,
		"route": _route("policies", tenant_id),
		"policies": service.list_policies(tenant_id),
		"theme": service.describe(tenant_id)["theme"]["components"]["ranking_policy"],
	}


def governance_model(
	service: RecsService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or RecsService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"route": _route("settings", tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
		"permissions": sorted({route["permission"] for route in contract["ui"]["routes"]}),
	}


def _route(name: str, tenant_id: str) -> dict[str, str]:
	for route in capability_routes(tenant_id):
		if route["name"] == name:
			return route
	raise KeyError(f"recs_route_not_found:{name}")
