"""Framework-neutral view models for APG Digital Payments."""

from __future__ import annotations

try:
	from .capability_contract import get_capability_contract
	from .service import DigitalPaymentsService
except ImportError:  # pragma: no cover - supports direct file loading in tests
	from capability_contract import get_capability_contract  # type: ignore
	from service import DigitalPaymentsService  # type: ignore


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	"""Return APG Python UI route metadata."""
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: DigitalPaymentsService | None = None, tenant_id: str = "default") -> dict[str, object]:
	"""Return a compact dashboard view model."""
	service = service or DigitalPaymentsService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"routes": capability_routes(tenant_id),
		"theme": contract["theme"],
	}


def order_console_model(service: DigitalPaymentsService, tenant_id: str) -> dict[str, object]:
	"""Return payment-order workbench state."""
	return {
		"tenant_id": tenant_id,
		"orders": service.list_orders(tenant_id),
		"evidence": service.list_evidence(tenant_id),
		"routes": capability_routes(tenant_id),
	}


def rule_console_model(tenant_id: str = "default") -> dict[str, object]:
	"""Return deterministic rule metadata for governance UIs."""
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"agents": contract["configuration"]["agents"],
	}
