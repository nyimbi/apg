"""Framework-neutral view models for APG Know Your Customer."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import KnowYourCustomerService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import KnowYourCustomerService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: KnowYourCustomerService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": contract["ui"]["routes"], "theme": contract["theme"], "streaming": contract["streaming"]}


def profile_console_model(service: KnowYourCustomerService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "profiles": service.list_profiles(tenant_id), "actions": ["open_profile", "register_document", "record_screening", "score_risk", "record_decision", "register_kyc_agent"]}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "configuration": contract["configuration"]}
