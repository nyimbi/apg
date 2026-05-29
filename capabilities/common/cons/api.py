"""API helpers for the Consent and Privacy Management capability."""

from __future__ import annotations

from typing import Any

from .service import ConsService


SERVICE = ConsService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"purpose_count": summary["purpose_count"],
		"active_consent_count": summary["active_consent_count"],
		"open_request_count": summary["open_request_count"],
		"coverage": summary["coverage"],
	}


def publish_notice(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_notice(
		notice_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		version=str(payload.get("version") or "v1"),
		url=str(payload["url"]),
		language=str(payload.get("language") or "en"),
		purposes=[str(item) for item in payload.get("purposes", [])],
		published_by=str(payload["published_by"]),
	)


def create_purpose(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_purpose(
		purpose_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		legal_basis=str(payload.get("legal_basis") or ""),
		retention_policy=str(payload.get("retention_policy") or ""),
		notice_id=str(payload["notice_id"]),
		data_categories=[str(item) for item in payload.get("data_categories", [])],
	)


def capture_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.capture_consent(
		consent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		purpose_id=str(payload["purpose_id"]),
		notice_id=str(payload["notice_id"]),
		source=str(payload.get("source") or "api"),
		captured_by=str(payload["captured_by"]),
	)


def withdraw_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.withdraw_consent(
		consent_id=str(payload["consent_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload["actor"]),
	)


def update_preferences(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.update_preferences(
		profile_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		channels=dict(payload.get("channels") or {}),
		purposes=dict(payload.get("purposes") or {}),
		updated_by=str(payload["updated_by"]),
	)


def process_consent_gated_data(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.process_consent_gated_data(
		decision_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		purpose_id=str(payload["purpose_id"]),
	)


def submit_privacy_request(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.submit_privacy_request(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		request_type=str(payload["request_type"]),
		submitted_by=str(payload["submitted_by"]),
		identity_verified=bool(payload.get("identity_verified", False)),
		evidence_reference=str(payload.get("evidence_reference") or ""),
	)


def complete_privacy_request(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.complete_privacy_request(
		request_id=str(payload["request_id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		actor=str(payload["actor"]),
		resolution=str(payload["resolution"]),
	)


def privacy_state(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"summary": SERVICE.dashboard_summary(tenant_id),
		"purposes": SERVICE.list_purposes(tenant_id),
		"notices": SERVICE.list_notices(tenant_id),
		"consents": SERVICE.list_consents(tenant_id),
		"preferences": SERVICE.list_preferences(tenant_id),
		"requests": SERVICE.list_requests(tenant_id),
		"processing_decisions": SERVICE.list_processing_decisions(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
	}
