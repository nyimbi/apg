"""Process-local API helpers for APG Digital Lending."""

from __future__ import annotations

from typing import Any

try:
	from .capability_contract import get_capability_contract
	from .service import LendingService
except ImportError:  # pragma: no cover
	from capability_contract import get_capability_contract  # type: ignore
	from service import LendingService  # type: ignore


SERVICE = LendingService()


def service() -> LendingService:
	return SERVICE


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = get_capability_contract(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {"capability": contract["capability"], "display_name": contract["display_name"], "tenant_id": tenant_id, "route_count": len(contract["ui"]["routes"]), "rule_count": len(contract["rule_engine"]["rules"]), "application_count": summary["application_count"], "offer_count": summary["offer_count"], "streaming": summary["streaming"]}


def register_product(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_product(str(payload["product_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["product_id"]), str(payload.get("owner_id") or ""), str(payload.get("product_type") or "term_loan"), str(payload.get("currency") or ""), payload.get("min_amount", 0), payload.get("max_amount", 0), int(payload.get("min_term_days", 0)), int(payload.get("max_term_days", 0)), payload.get("annual_rate", 0), str(payload.get("repayment_frequency") or "monthly"), bool(payload.get("policy_attached", True)))


def onboard_borrower(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.onboard_borrower(str(payload["borrower_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("customer_reference") or ""), str(payload.get("kyc_profile_id") or ""), str(payload.get("country") or ""), str(payload.get("income_evidence_id") or ""), str(payload.get("consent_reference") or ""), bool(payload.get("policy_attached", True)))


def submit_application(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_application(str(payload["application_id"]), str(payload.get("tenant_id") or "default"), str(payload["borrower_id"]), str(payload["product_id"]), payload.get("requested_amount", 0), str(payload.get("purpose") or ""), str(payload.get("affordability_reference") or ""), str(payload.get("bank_statement_reference") or ""), str(payload.get("aml_reference") or ""), str(payload.get("fraud_reference") or ""), str(payload.get("behavior_evidence_reference") or ""), str(payload.get("human_review") or ""), bool(payload.get("policy_attached", True)))


def record_underwriting(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_underwriting(str(payload["underwriting_id"]), str(payload.get("tenant_id") or "default"), str(payload["application_id"]), payload.get("score", 0), str(payload.get("decision") or "refer"), list(payload.get("evidence_references") or []), str(payload.get("human_approval") or ""), str(payload.get("adverse_reason") or ""))


def issue_offer(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.issue_offer(str(payload["offer_id"]), str(payload.get("tenant_id") or "default"), str(payload["application_id"]), str(payload["underwriting_id"]), payload.get("amount", 0), payload.get("apr", 0), int(payload.get("term_days", 0)), str(payload.get("expiry_date") or ""), str(payload.get("status") or "issued"), str(payload.get("borrower_acceptance_reference") or ""))


def record_disbursement(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_disbursement(str(payload["disbursement_id"]), str(payload.get("tenant_id") or "default"), str(payload["offer_id"]), payload.get("amount", 0), str(payload.get("rail") or "payment_account"), str(payload.get("funding_account") or ""), str(payload.get("destination_reference") or ""), str(payload.get("human_approval") or ""))


def schedule_repayment(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.schedule_repayment(str(payload["schedule_id"]), str(payload.get("tenant_id") or "default"), str(payload["offer_id"]), payload.get("due_amount", 0), str(payload.get("due_date") or ""), str(payload.get("frequency") or "monthly"), int(payload.get("installment_count", 1)))


def open_collection_case(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.open_collection_case(str(payload["case_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("overdue_account_reference") or ""), str(payload.get("reason") or ""), str(payload.get("reviewer_id") or ""), str(payload.get("contact_policy_reference") or ""))


def register_lending_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_lending_agent(str(payload["agent_id"]), str(payload.get("tenant_id") or "default"), str(payload.get("name") or payload["agent_id"]), str(payload.get("runtime") or "codex"), str(payload.get("role") or "lending_ops_reviewer"), str(payload.get("scope") or "review lending operations"))


def list_applications(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_applications(tenant_id)
