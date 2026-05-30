"""UI metadata helpers for the Wallet and Payment Core capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import WaltService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def wallet_console_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/wallets",
		"tenant_id": tenant_id,
		"wallets": service.list_wallets(tenant_id),
		"wallet_statuses": ["active", "disabled", "frozen"],
		"multi_currency_supported": True,
	}


def transaction_console_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/transactions",
		"tenant_id": tenant_id,
		"transactions": service.list_transactions(tenant_id),
		"transaction_statuses": ["authorized", "captured", "review_required", "declined", "settled"],
		"high_value_mfa_required": True,
		"streaming": service.describe(tenant_id)["streaming"],
	}


def instrument_vault_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/instruments",
		"tenant_id": tenant_id,
		"instruments": service.list_instruments(tenant_id),
		"instrument_types": ["card", "bank_account", "mobile_money", "token", "external"],
		"tokenization_required": True,
		"encryption_required": True,
	}


def settlement_center_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/settlement",
		"tenant_id": tenant_id,
		"settlement_batches": service.list_settlement_batches(tenant_id),
		"settlement_statuses": ["ready", "reconciled", "exception_review"],
		"reconciliation_required": True,
		"streaming": service.describe(tenant_id)["streaming"],
	}


def reconciliation_queue_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/reconciliation",
		"tenant_id": tenant_id,
		"reconciliations": service.list_reconciliations(tenant_id),
		"exception_batches": [
			batch
			for batch in service.list_settlement_batches(tenant_id)
			if batch["status"] == "exception_review"
		],
	}


def risk_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	return {
		"route": "/walt/risk",
		"tenant_id": tenant_id,
		"review_required_transactions": [
			transaction
			for transaction in service.list_transactions(tenant_id)
			if transaction["status"] == "review_required"
		],
		"rules": get_capability_contract(tenant_id)["rule_engine"]["rules"],
	}


def agent_workbench_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	contract = service.describe(tenant_id)
	return {
		"route": "/walt/agents",
		"tenant_id": tenant_id,
		"agents": service.list_walt_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["walt_agents"]["supported_runtimes"],
		"supported_roles": contract["configuration"]["walt_agents"]["supported_roles"],
		"human_approval_required": contract["configuration"]["walt_agents"]["human_approval_required"],
	}


def policy_center_model(
	service: WaltService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or WaltService()
	contract = service.describe(tenant_id)
	return {
		"route": "/walt/policy",
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"review_required_transactions": [
			transaction
			for transaction in service.list_transactions(tenant_id)
			if transaction["status"] == "review_required"
		],
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"route": "/walt/settings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}
