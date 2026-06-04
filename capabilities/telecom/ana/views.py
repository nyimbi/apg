"""View models for APG Telecom Analytics screens."""

from __future__ import annotations

from typing import Any

from .capability_contract import get_capability_contract
from .service import TelecomAnaService


def dashboard_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": "Telecom Analytics", "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "theme": contract["theme"], "routes": contract["ui"]["routes"]}


def analysis_console_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "analysis_runs": _items(service.analysis_runs, tenant_id), "metrics": _items(service.metrics, tenant_id), "churn_predictions": _items(service.churn_predictions, tenant_id)}


def revenue_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "revenue_events": _items(service.revenue_events, tenant_id), "anomalies": [i.to_dict() for i in service.anomalies.values() if i.tenant_id == tenant_id and i.anomaly_type == "revenue_leak"]}


def network_analytics_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "network_analytics": _items(service.network_analytics, tenant_id), "anomalies": _items(service.anomalies, tenant_id)}


def model_registry_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "models": _items(service.models, tenant_id), "supported_model_types": contract["configuration"]["models"]["supported_model_types"]}


def agent_workbench_model(service: TelecomAnaService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "supported_runtimes": contract["configuration"]["agents"]["supported_runtimes"], "supported_roles": contract["configuration"]["agents"]["supported_roles"], "agents": _items(service.agents, tenant_id)}


def _items(store: dict[Any, Any], tenant_id: str) -> list[dict[str, Any]]:
	return [item.to_dict() for item in sorted(store.values(), key=lambda v: v.id) if item.tenant_id == tenant_id]
