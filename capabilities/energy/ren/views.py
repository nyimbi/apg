"""View models for Renewable Energy screens."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RenewableEnergyService
except ImportError:
	from capability_contract import get_capability_contract  # type: ignore
	from service import RenewableEnergyService  # type: ignore


def dashboard_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"title": "Renewable Energy",
		"tenant_id": tenant_id,
		"summary": svc.dashboard_summary(tenant_id),
		"theme": contract["theme"],
		"routes": contract["ui"]["routes"],
	}


def asset_registry_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"assets": svc.list_assets(tenant_id),
		"supported_types": contract["configuration"]["assets"]["supported_types"],
		"supported_statuses": contract["configuration"]["assets"]["supported_statuses"],
	}


def asset_detail_model(svc: RenewableEnergyService, tenant_id: str, asset_id: str) -> dict[str, Any]:
	asset = svc.get_asset(tenant_id, asset_id)
	curtailments = svc.list_curtailments(tenant_id, asset_id=asset_id)
	recs = svc.list_recs(tenant_id)
	asset_recs = [r for r in recs if r["asset_id"] == asset_id]
	forecasts = svc.list_forecasts(tenant_id, asset_id=asset_id)
	metrics = svc.list_performance_metrics(tenant_id, asset_id=asset_id)
	fits = svc.list_fits(tenant_id, asset_id=asset_id)
	return {
		"tenant_id": tenant_id,
		"asset": asset,
		"curtailments": curtailments,
		"rec_certificates": asset_recs,
		"recent_forecasts": forecasts[-5:],
		"performance_metrics": metrics,
		"feed_in_tariffs": fits,
	}


def curtailment_tracker_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"curtailments": svc.list_curtailments(tenant_id),
		"summary": svc.get_curtailment_summary(tenant_id),
		"supported_reasons": contract["configuration"]["curtailment"]["supported_reasons"],
	}


def rec_manager_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rec_certificates": svc.list_recs(tenant_id),
		"issued_recs": svc.list_recs(tenant_id, status="issued"),
		"supported_types": contract["configuration"]["recs"]["supported_types"],
		"supported_statuses": contract["configuration"]["recs"]["supported_statuses"],
	}


def carbon_credit_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"carbon_credits": svc.list_carbon_credits(tenant_id),
		"issued_credits": svc.list_carbon_credits(tenant_id, status="issued"),
		"supported_types": contract["configuration"]["carbon_credits"]["supported_types"],
	}


def fit_manager_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"feed_in_tariffs": svc.list_fits(tenant_id),
		"active_fits": [f for f in svc.list_fits(tenant_id) if f["status"] == "active"],
		"supported_types": contract["configuration"]["feed_in_tariffs"]["supported_types"],
	}


def forecasting_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"forecasts": svc.list_forecasts(tenant_id),
		"supported_types": contract["configuration"]["forecasting"]["supported_types"],
		"supported_horizons": contract["configuration"]["forecasting"]["supported_horizons"],
	}


def performance_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"metrics": svc.list_performance_metrics(tenant_id),
		"assets": svc.list_assets(tenant_id),
		"supported_metrics": contract["configuration"]["performance"]["supported_metrics"],
	}


def agent_workbench_model(svc: RenewableEnergyService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": _tenant_items(svc.agents, tenant_id),
		"supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["agents"]["supported_roles"],
	}


def _tenant_items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [v.to_dict() for k, v in store.items() if k[0] == tenant_id]
