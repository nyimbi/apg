"""Framework-neutral view models for APG Digital Lending."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import LendingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import LendingService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: LendingService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": contract["ui"]["routes"], "theme": contract["theme"], "streaming": contract["streaming"]}


def lending_console_model(service: LendingService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "applications": service.list_applications(tenant_id), "offers": service.list_offers(tenant_id), "repayments": service.list_repayments(tenant_id), "actions": ["register_product", "onboard_borrower", "submit_application", "record_underwriting", "issue_offer", "record_disbursement", "schedule_repayment", "open_collection_case", "register_lending_agent"]}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "configuration": contract["configuration"]}
