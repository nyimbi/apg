"""Framework-neutral view models for APG Digital Wallets."""

from __future__ import annotations

try:
	from .capability_contract import get_capability_contract
	from .service import DigitalWalletsService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import DigitalWalletsService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: DigitalWalletsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or DigitalWalletsService()
	contract = service.describe(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "summary": service.dashboard_summary(tenant_id), "routes": capability_routes(tenant_id), "theme": contract["theme"]}


def wallet_console_model(service: DigitalWalletsService, tenant_id: str) -> dict[str, object]:
	return {"tenant_id": tenant_id, "wallets": service.list_wallets(tenant_id), "ledger": service.list_ledger(tenant_id), "routes": capability_routes(tenant_id)}


def rule_console_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {"tenant_id": tenant_id, "rules": contract["rule_engine"]["rules"], "streaming": contract["streaming"], "agents": contract["configuration"]["agents"]}
