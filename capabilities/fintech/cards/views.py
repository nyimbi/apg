"""Framework-neutral view models for APG Digital Cards."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import CardService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import CardService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: CardService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": contract["ui"]["routes"], "theme": contract["theme"], "streaming": contract["streaming"]}


def card_console_model(service: CardService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "cards": service.list_cards(tenant_id), "authorizations": service.list_authorizations(tenant_id), "actions": ["register_program", "onboard_cardholder", "issue_card", "provision_token", "authorize_transaction", "file_dispute", "register_card_agent"]}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "configuration": contract["configuration"]}
