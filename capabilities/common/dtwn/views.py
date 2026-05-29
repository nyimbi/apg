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
