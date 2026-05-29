"""UI metadata and view models for Predictive Analytics."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import PredService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def forecast_console_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"forecasts": service.list_forecasts(tenant_id),
		"models": service.list_models(tenant_id),
		"route": "/pred/forecasts",
	}


def score_monitor_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"scores": service.list_scores(tenant_id),
		"feature_sets": service.list_feature_sets(tenant_id),
		"drift_reports": service.list_drift_reports(tenant_id),
		"route": "/pred/scores",
	}


def scenario_lab_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"scenarios": service.list_scenarios(tenant_id),
		"models": service.list_models(tenant_id),
		"route": "/pred/scenarios",
	}


def model_board_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"models": service.list_models(tenant_id),
		"feature_sets": service.list_feature_sets(tenant_id),
		"route": "/pred/models",
	}


def governance_model(
	service: PredService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or PredService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"drift_reports": service.list_drift_reports(tenant_id),
		"rules": service.describe(tenant_id)["rule_engine"]["rules"],
		"route": "/pred/governance",
	}
