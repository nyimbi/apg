"""API helpers for APG Blockchain Ledger Services."""

from __future__ import annotations

from typing import Any

from .service import BclgService


SERVICE = BclgService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.ledger_summary(tenant_id),
	}


def register_ledger(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_ledger(
		ledger_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		consensus_profile=str(payload.get("consensus_profile") or ""),
		network_policy=str(payload.get("network_policy") or ""),
		participants=[str(item) for item in payload.get("participants", [])],
		fork_monitoring_enabled=_payload_bool(payload, "fork_monitoring_enabled", True),
	)


def bind_key_custody(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.bind_key_custody(
		binding_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ledger_id=str(payload["ledger_id"]),
		key_id=str(payload["key_id"]),
		custodian=str(payload.get("custodian") or "key-custodian"),
		rotation_policy=str(payload.get("rotation_policy") or "90d"),
	)


def submit_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_transaction(
		transaction_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ledger_id=str(payload["ledger_id"]),
		from_account=str(payload["from_account"]),
		to_account=str(payload["to_account"]),
		amount=float(payload["amount"]),
		asset=str(payload.get("asset") or "TOKEN"),
		signature=str(payload.get("signature") or ""),
		key_custody_id=str(payload.get("key_custody_id") or ""),
		compliance_tags=[str(item) for item in payload.get("compliance_tags", [])],
		transaction_review_recorded=_payload_bool(payload, "transaction_review_recorded", False),
		actor=str(payload.get("actor") or "ledger-operator"),
	)


def request_transaction_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_transaction_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		transaction_id=str(payload["transaction_id"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
	)


def decide_transaction_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_transaction_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def approve_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_transaction(
		transaction_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
		tenant_id=str(payload["tenant_id"]) if payload.get("tenant_id") else None,
		notes=str(payload.get("notes") or "Approved transaction review."),
	)


def request_contract_deployment_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_contract_deployment_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ledger_id=str(payload["ledger_id"]),
		name=str(payload.get("name") or payload["id"]),
		version=str(payload.get("version") or "1.0.0"),
		artifact_hash=str(payload["artifact_hash"]),
		requested_by=str(payload["requested_by"]),
		rollback_plan=str(payload["rollback_plan"]),
	)


def decide_contract_deployment_approval(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_contract_deployment_approval(
		approval_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def deploy_contract(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deploy_contract(
		contract_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ledger_id=str(payload["ledger_id"]),
		name=str(payload.get("name") or payload["id"]),
		version=str(payload.get("version") or "1.0.0"),
		artifact_hash=str(payload.get("artifact_hash") or ""),
		reviewed_by=str(payload.get("reviewed_by") or ""),
		rollback_plan=str(payload.get("rollback_plan") or ""),
		approval_id=str(payload["approval_id"]) if payload.get("approval_id") else None,
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_ledgers(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_ledgers(tenant_id)


def list_key_custody(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_key_custody(tenant_id)


def list_transactions(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_transactions(tenant_id)


def list_transaction_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_transaction_reviews(tenant_id)


def list_contracts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_contracts(tenant_id)


def list_contract_deployment_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_contract_deployment_approvals(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
