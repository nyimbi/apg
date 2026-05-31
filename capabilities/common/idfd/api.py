"""API helpers for the APG Identity Federation capability."""

from __future__ import annotations

from typing import Any

from .service import IdfdService


SERVICE = IdfdService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"provider_count": summary["provider_count"],
		"active_session_count": summary["active_session_count"],
		"stale_provider_count": summary["stale_provider_count"],
		"federation_agent_count": summary["federation_agent_count"],
		"lifecycle_batch_count": summary["lifecycle_batch_count"],
	}


def register_provider(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_provider(
		provider_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		protocol=str(payload["protocol"]),
		owner_id=str(payload.get("owner_id") or ""),
		signing_key_id=str(payload.get("signing_key_id") or ""),
		metadata_url=str(payload.get("metadata_url") or ""),
		assertion_encrypted=bool(payload.get("assertion_encrypted", True)),
		redirect_allowlist=list(payload.get("redirect_allowlist") or []),
		pkce_required=bool(payload.get("pkce_required", True)),
		metadata_refresh_completed=bool(payload.get("metadata_refresh_completed", True)),
		metadata_age_hours=float(payload.get("metadata_age_hours") or 0),
		status=str(payload.get("status") or "active"),
	)


def refresh_provider_metadata(provider_id: str, tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.refresh_provider_metadata(provider_id, tenant_id)


def add_claim_mapping(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_claim_mapping(
		mapping_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		provider_id=str(payload["provider_id"]),
		source_claim=str(payload["source_claim"]),
		target_claim=str(payload["target_claim"]),
		transform=str(payload.get("transform") or "copy"),
		reviewed=bool(payload.get("reviewed", True)),
	)


def issue_session(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.issue_session(
		session_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		provider_id=str(payload["provider_id"]),
		subject_id=str(payload["subject_id"]),
		session_privilege=str(payload.get("session_privilege") or "standard"),
		mfa_completed=bool(payload.get("mfa_completed", True)),
		risk_score=float(payload.get("risk_score") or 0),
		max_session_hours=int(payload["max_session_hours"]) if payload.get("max_session_hours") is not None else None,
	)


def revoke_session(session_id: str, tenant_id: str = "default", reason: str = "manual") -> dict[str, Any]:
	return SERVICE.revoke_session(session_id, tenant_id, reason)


def register_certificate(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_certificate(
		certificate_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		provider_id=str(payload["provider_id"]),
		key_id=str(payload["key_id"]),
		expires_at=str(payload["expires_at"]),
		active=bool(payload.get("active", True)),
		rotated_at=payload.get("rotated_at"),
	)


def health_report(report_id: str, tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.health_report(report_id, tenant_id)


def register_federation_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_federation_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_idfd_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count", 1)),
		operation=str(payload.get("operation") or "federation_agent_batch"),
		batch_id=payload.get("id"),
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


def list_federation_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_federation_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
