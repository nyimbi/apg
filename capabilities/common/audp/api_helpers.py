"""Dependency-light AUDP API helpers for generated APG applications."""

from __future__ import annotations

from typing import Any

from .audio_runtime import AudpService


SERVICE = AudpService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**SERVICE.audio_summary(tenant_id),
	}


def record_consent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.record_consent(
		consent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		consent_type=str(payload["consent_type"]),
		subject_id=str(payload["subject_id"]),
		granted_by=str(payload["granted_by"]),
		evidence=str(payload["evidence"]),
		scope=dict(payload.get("scope") or {}),
	)


def attach_model_policy(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.attach_model_policy(
		policy_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		model_id=str(payload["model_id"]),
		policy_name=str(payload["policy_name"]),
		allowed_operations=[str(item) for item in payload["allowed_operations"]],
		attached_by=str(payload["attached_by"]),
		risk_tier=str(payload.get("risk_tier") or "standard"),
	)


def request_transcription(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_transcription(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		audio_source_id=str(payload["audio_source_id"]),
		requested_by=str(payload["requested_by"]),
		model_id=str(payload["model_id"]),
		language_code=str(payload.get("language_code") or "auto"),
		confidence=float(payload.get("confidence", 1.0)),
		retention_policy=str(payload.get("retention_policy") or "default"),
		result=dict(payload.get("result") or {}),
		human_review_recorded=_payload_bool(payload, "human_review_recorded", False),
	)


def decide_transcript_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_transcript_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def request_synthesis(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_synthesis(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		text=str(payload["text"]),
		requested_by=str(payload["requested_by"]),
		model_id=str(payload["model_id"]),
		watermark_applied=_payload_bool(payload, "watermark_applied", True),
		retention_policy=str(payload.get("retention_policy") or "default"),
	)


def decide_synthesis_review(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.decide_synthesis_review(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload["notes"]),
	)


def request_voice_clone(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_voice_clone(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		voice_owner_id=str(payload["voice_owner_id"]),
		requested_by=str(payload["requested_by"]),
		model_id=str(payload["model_id"]),
		retention_policy=str(payload.get("retention_policy") or "default"),
	)


def request_analysis(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.request_analysis(
		job_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		audio_source_id=str(payload["audio_source_id"]),
		requested_by=str(payload["requested_by"]),
		model_id=str(payload["model_id"]),
		analysis_types=[str(item) for item in payload["analysis_types"]],
		retention_policy=str(payload.get("retention_policy") or "default"),
	)


def register_audio_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_audio_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime=str(payload["runtime"]),
		role=str(payload["role"]),
		scope=str(payload["scope"]),
		contribution_disclosed=_payload_bool(payload, "contribution_disclosed", True),
		policy_ref=str(payload.get("policy_ref") or ""),
		registered=_payload_bool(payload, "registered", True),
	)


def change_job_state(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.change_job_state(
		tenant_id=str(payload.get("tenant_id") or "default"),
		job_id=str(payload["id"]),
		status=str(payload["status"]),
		reason=str(payload["reason"]),
		audit_recorded=_payload_bool(payload, "audit_recorded", True),
	)


def list_consents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_consents(tenant_id)


def list_model_policies(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_model_policies(tenant_id)


def list_jobs(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_jobs(tenant_id)


def list_transcript_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_transcript_reviews(tenant_id)


def list_synthesis_reviews(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_synthesis_reviews(tenant_id)


def list_audio_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_audio_agents(tenant_id)


def list_governance_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_governance_events(tenant_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "completed"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def _payload_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
	value = payload.get(key, default)
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "on"}
	return bool(value)
