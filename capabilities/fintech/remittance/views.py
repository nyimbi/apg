"""Framework-neutral view models for APG Cross-Border Remittance."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import RemittanceService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import RemittanceService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, Any]]:
	return get_capability_contract(tenant_id)["ui"]["routes"]


def dashboard_model(service: RemittanceService, tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"title": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": contract["ui"]["routes"], "theme": contract["theme"], "streaming": contract["streaming"]}


def transfer_console_model(service: RemittanceService, tenant_id: str = "default") -> dict[str, Any]:
	return {"tenant_id": tenant_id, "quotes": service.list_quotes(tenant_id), "transfers": service.list_transfers(tenant_id), "actions": ["create_quote", "create_transfer", "release_payout", "file_refund", "register_remittance_agent"]}


def rule_console_model(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "configuration": contract["configuration"]}
