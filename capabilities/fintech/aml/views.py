"""Framework-neutral view models for APG Anti Money Laundering."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AntiMoneyLaunderingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import AntiMoneyLaunderingService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: AntiMoneyLaunderingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": contract["ui"]["routes"], "theme": contract["theme"], "streaming": contract["streaming"]}


def alert_console_model(service: AntiMoneyLaunderingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "alerts": service.list_alerts(tenant_id), "cases": service.list_cases(tenant_id), "actions": ["monitor_transaction", "create_alert", "triage_alert", "open_case", "draft_sar", "register_aml_agent"]}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "configuration": contract["configuration"]}
