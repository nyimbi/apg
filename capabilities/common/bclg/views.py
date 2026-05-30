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
	service = _service_or_default(service)
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
		"transaction_reviews": service.list_transaction_reviews(tenant_id),
		"contracts": service.list_contracts(tenant_id),
		"contract_reviews": service.list_contract_deployment_approvals(tenant_id),
		"key_custody": service.list_key_custody(tenant_id),
		"ledger_agents": service.list_ledger_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"review_queue": [
			item for item in service.list_transactions(tenant_id)
			if item["status"] == "pending_review"
		],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def ledger_console_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
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
	service = _service_or_default(service)
	transactions = service.list_transactions(tenant_id)
	return {
		"tenant_id": tenant_id,
		"transactions": transactions,
		"reviews": service.list_transaction_reviews(tenant_id),
		"pending_review": [item for item in transactions if item["status"] == "pending_review"],
		"committed": [item for item in transactions if item["status"] == "committed"],
		"rejected": [item for item in transactions if item["status"] == "rejected"],
	}


def transaction_review_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	reviews = service.list_transaction_reviews(tenant_id)
	return {
		"tenant_id": tenant_id,
		"reviews": reviews,
		"pending_reviews": [item for item in reviews if item["status"] == "pending"],
		"decided_reviews": [item for item in reviews if item["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["independent_reviewer", "reviewer_notes_required", "matching_transaction_required"],
	}


def contract_registry_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	return {
		"tenant_id": tenant_id,
		"contracts": service.list_contracts(tenant_id),
		"approvals": service.list_contract_deployment_approvals(tenant_id),
		"required_deployment_fields": ["approval_id", "artifact_hash", "rollback_plan"],
	}


def contract_review_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	approvals = service.list_contract_deployment_approvals(tenant_id)
	return {
		"tenant_id": tenant_id,
		"approvals": approvals,
		"pending_approvals": [item for item in approvals if item["status"] == "pending"],
		"decided_approvals": [item for item in approvals if item["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["independent_reviewer", "artifact_hash_required", "rollback_plan_required"],
	}


def audit_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": service.ledger_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def ledger_agent_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": service.list_ledger_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["ledger_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["ledger_agents"]["allowed_roles"],
		"required_fields": ["name", "runtime", "role", "scope", "contribution_disclosed"],
		"actions": ["register", "scope", "review_contribution", "deactivate"],
	}


def analytics_model(
	service: BclgService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = _service_or_default(service)
	summary = service.ledger_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": summary,
		"commit_rate": _safe_ratio(summary["committed_transaction_count"], summary["transaction_count"]),
		"review_rate": _safe_ratio(summary["transaction_review_count"], summary["transaction_count"]),
		"deployment_rate": _safe_ratio(summary["deployed_contract_count"], summary["contract_count"]),
		"agent_coverage": _safe_ratio(summary["ledger_agent_count"], max(summary["ledger_count"], 1)),
	}


def settings_model(tenant_id: str = "default") -> dict[str, object]:
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"rules": contract["rule_engine"]["rules"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
	}


def _service_or_default(service: BclgService | None) -> BclgService:
	if service is not None:
		return service
	try:
		from .api import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return BclgService()


def _safe_ratio(numerator: int, denominator: int) -> float:
	return round(numerator / denominator, 4) if denominator else 0.0
