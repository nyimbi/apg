"""Dependency-light BIOP API helpers for generated APG applications."""

from __future__ import annotations

from typing import Any

from .biometric_runtime import BiopService


SERVICE = BiopService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		**SERVICE.biometric_summary(tenant_id),
	}


def record_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_consent(
		consent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		purpose=str(payload["purpose"]),
		modalities=[str(item) for item in payload["modalities"]],
		jurisdictions=[str(item) for item in payload["jurisdictions"]],
		granted_by=str(payload["granted_by"]),
		evidence=str(payload["evidence"]),
	)


def revoke_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.revoke_consent(
		consent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		revoked_by=str(payload["revoked_by"]),
		reason=str(payload["reason"]),
	)


def enroll_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.enroll_template(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		modality=str(payload["modality"]),
		template_hash=str(payload["template_hash"]),
		encrypted=_payload_bool(payload, "encrypted", False),
		quality_score=float(payload["quality_score"]),
		consent_id=str(payload["consent_id"]),
		retention_policy=str(payload["retention_policy"]),
	)


def retire_template(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.retire_template(
		template_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		retired_by=str(payload["retired_by"]),
		reason=str(payload["reason"]),
	)


def request_verification(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_verification(
		verification_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_id=str(payload["subject_id"]),
		template_id=str(payload["template_id"]),
		modality=str(payload["modality"]),
		requested_by=str(payload["requested_by"]),
		match_confidence=float(payload["match_confidence"]),
		liveness_score=float(payload["liveness_score"]),
		source_jurisdiction=str(payload.get("source_jurisdiction") or ""),
		target_jurisdiction=str(payload.get("target_jurisdiction") or ""),
		privacy_review_recorded=_payload_bool(payload, "privacy_review_recorded", False),
		human_review_recorded=_payload_bool(payload, "human_review_recorded", False),
	)


def request_privacy_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_privacy_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		verification_id=str(payload["verification_id"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
	)


def decide_privacy_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_privacy_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def request_match_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_match_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		verification_id=str(payload["verification_id"]),
		requested_by=str(payload["requested_by"]),
		justification=str(payload["justification"]),
	)


def decide_match_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_match_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def register_biometric_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_biometric_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		owner=str(payload["owner"]),
		purpose=str(payload["purpose"]),
		contribution_disclosed=_payload_bool(payload, "contribution_disclosed", True),
		human_approval_required=_payload_bool(payload, "human_approval_required", False),
	)


def validate_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_biop_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or "bytewax"),
		mutation_count=int(payload.get("mutation_count") or 1),
		operation=str(payload.get("operation") or "biometric_agent_batch"),
		batch_id=payload.get("id"),
	)


def list_consents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_consents(tenant_id)


def list_templates(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_templates(tenant_id)


def list_verifications(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_verifications(tenant_id)


def list_reviews(tenant_id: str | None = None, review_type: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_reviews(tenant_id, review_type)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audit_events(tenant_id)


def list_biometric_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_biometric_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
