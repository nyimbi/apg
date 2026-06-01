"""View models for APG Agency Banking."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import AgencyBankingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import AgencyBankingService  # type: ignore


def dashboard_model(service: AgencyBankingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = service.dashboard_summary(tenant_id)
	return {
		"title": "Agency Banking",
		"tenant_id": tenant_id,
		"summary": summary,
		"cards": [
			{"label": "Programs", "value": summary["program_count"], "icon": "network"},
			{"label": "Outlets", "value": summary["outlet_count"], "icon": "store"},
			{"label": "Agents", "value": summary["agent_count"], "icon": "badge-check"},
			{"label": "Float Accounts", "value": summary["float_account_count"], "icon": "wallet"},
			{"label": "Transactions", "value": summary["transaction_count"], "icon": "receipt"},
			{"label": "Supervision", "value": summary["supervision_count"], "icon": "clipboard-check"},
		],
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
	}


def agency_console_model(service: AgencyBankingService, tenant_id: str) -> dict[str, Any]:
	return {
		"tenant_id": tenant_id,
		"programs": [item.to_dict() for item in sorted(service.programs.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"outlets": service.list_outlets(tenant_id),
		"agents": [item.to_dict() for item in sorted(service.agents.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"float_accounts": service.list_float_accounts(tenant_id),
		"customers": [item.to_dict() for item in sorted(service.customers.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"transactions": service.list_transactions(tenant_id),
		"cash_movements": [item.to_dict() for item in sorted(service.cash_movements.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"commissions": [item.to_dict() for item in sorted(service.commissions.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"disputes": [item.to_dict() for item in sorted(service.disputes.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"supervision": [item.to_dict() for item in sorted(service.supervision_visits.values(), key=lambda item: item.id) if item.tenant_id == tenant_id],
		"ai_agents": [item.to_dict() for item in sorted(service.evidence.values(), key=lambda item: item.id) if item.tenant_id == tenant_id and item.kind == "ai_agent"],
	}


def route_models(service: AgencyBankingService, tenant_id: str) -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	console = agency_console_model(service, tenant_id)
	return {route["name"]: {"route": route["path"], "component": route["component"], "permission": route["permission"], "data": console if route["name"] != "dashboard" else dashboard_model(service, tenant_id)} for route in contract["ui"]["routes"]}
