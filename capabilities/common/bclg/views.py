"""UI metadata helpers for APG Blockchain Ledger Services."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import BclgService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BclgService()
	contract = service.describe(tenant_id)
	summary = service.ledger_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": summary,
		"ledgers": service.list_ledgers(tenant_id),
		"transactions": service.list_transactions(tenant_id),
		"contracts": service.list_contracts(tenant_id),
		"key_custody": service.list_key_custody(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"review_queue": [
			item for item in service.list_transactions(tenant_id)
			if item["status"] == "pending_review"
		],
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def ledger_console_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BclgService()
	return {
		"tenant_id": tenant_id,
		"ledgers": service.list_ledgers(tenant_id),
		"key_custody": service.list_key_custody(tenant_id),
		"routes": [route for route in capability_routes(tenant_id) if route["nav_group"] == "Ledgers"],
	}


def transaction_monitor_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or BclgService()
	transactions = service.list_transactions(tenant_id)
	return {
		"tenant_id": tenant_id,
		"transactions": transactions,
		"pending_review": [item for item in transactions if item["status"] == "pending_review"],
		"committed": [item for item in transactions if item["status"] == "committed"],
	}
