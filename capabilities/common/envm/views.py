"""UI metadata helpers for the Environment Management capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import EnvmService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: EnvmService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or EnvmService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"records": service.list_records(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}
