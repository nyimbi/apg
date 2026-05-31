"""API helpers for the Encryption Services capability."""

from __future__ import annotations

from typing import Any

from .service import EncrService


SERVICE = EncrService()


def _required_tenant_id(payload: dict[str, Any]) -> str:
	tenant_id = str(payload.get("tenant_id") or "").strip()
	if not tenant_id:
		raise PermissionError("tenant_context_required")
	return tenant_id


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"key_domain_count": summary["key_domain_count"],
		"operation_count": summary["operation_count"],
		"crypto_agent_count": summary["crypto_agent_count"],
		"denied_operation_count": summary["denied_operation_count"],
		"review_required_count": summary["review_required_count"],
		"pending_exception_count": summary["pending_exception_count"],
		"scheduled_rotation_count": summary["scheduled_rotation_count"],
	}


def register_key_domain(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_key_domain(
		tenant_id=_required_tenant_id(payload),
		domain_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		algorithm=str(payload.get("algorithm") or "AES-256-GCM"),
		data_classification=str(payload.get("data_classification") or "confidential"),
		entropy_quality=payload.get("entropy_quality", 0.99),
	)


def evaluate_crypto_operation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.evaluate_crypto_operation(
		tenant_id=_required_tenant_id(payload),
		operation_id=str(payload["id"]),
		operation_type=str(payload.get("operation_type") or "encrypt"),
		key_domain_id=str(payload["key_domain_id"]),
		data_classification=payload.get("data_classification"),
		algorithm=payload.get("algorithm"),
		algorithm_family=payload.get("algorithm_family"),
		entropy_quality=payload.get("entropy_quality"),
		plaintext_export_requested=bool(payload.get("plaintext_export_requested", False)),
		active_threat_signal=bool(payload.get("active_threat_signal", False)),
	)


def request_crypto_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_crypto_exception(
		tenant_id=_required_tenant_id(payload),
		review_id=str(payload["id"]),
		operation_id=str(payload["operation_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
	)


def decide_crypto_exception(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_crypto_exception(
		tenant_id=_required_tenant_id(payload),
		review_id=str(payload["id"]),
		reviewer=str(payload.get("reviewer") or ""),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def schedule_key_rotation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.schedule_key_rotation(
		tenant_id=_required_tenant_id(payload),
		rotation_id=str(payload["id"]),
		key_domain_id=str(payload["key_domain_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		reason=str(payload.get("reason") or ""),
	)


def complete_key_rotation(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_key_rotation(
		tenant_id=_required_tenant_id(payload),
		rotation_id=str(payload["id"]),
		actor=str(payload.get("actor") or ""),
		evidence=str(payload.get("evidence") or ""),
	)


def register_crypto_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_crypto_agent(
		tenant_id=_required_tenant_id(payload),
		agent_id=str(payload["id"]),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or ""),
		purpose=str(payload.get("purpose") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
		policy_ref=payload.get("policy_ref"),
	)


def validate_crypto_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_crypto_lifecycle_batch(
		tenant_id=_required_tenant_id(payload),
		event_stream=str(payload.get("event_stream") or ""),
		mutation_count=int(payload.get("mutation_count") or 0),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=_required_tenant_id(payload),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_crypto_posture(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"key_domains": SERVICE.list_key_domains(tenant_id),
		"operations": SERVICE.list_operations(tenant_id),
		"exception_reviews": SERVICE.list_exception_reviews(tenant_id),
		"rotations": SERVICE.list_rotations(tenant_id),
		"crypto_agents": SERVICE.list_crypto_agents(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}
