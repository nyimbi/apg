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
		fork_monitoring_enabled=bool(payload.get("fork_monitoring_enabled", True)),
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
		transaction_review_recorded=bool(payload.get("transaction_review_recorded", False)),
		actor=str(payload.get("actor") or "ledger-operator"),
	)


def approve_transaction(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.approve_transaction(
		transaction_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or "reviewer"),
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


def list_contracts(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_contracts(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)
