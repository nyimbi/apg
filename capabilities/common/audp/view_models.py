"""Dependency-light AUDP view models for package-composed UIs."""

from __future__ import annotations

from .audio_runtime import AudpService
from .capability_contract import get_capability_contract


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	contract = get_capability_contract(tenant_id)
	return [
		{
			"name": route["name"],
			"path": route["path"],
			"component": route["component"],
			"permission": route["permission"],
			"nav_group": route["nav_group"],
		}
		for route in contract["ui"]["routes"]
	]


def dashboard_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	contract = get_capability_contract(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.audio_summary(tenant_id),
		"consents": service.list_consents(tenant_id),
		"model_policies": service.list_model_policies(tenant_id),
		"jobs": service.list_jobs(tenant_id),
		"transcript_reviews": service.list_transcript_reviews(tenant_id),
		"synthesis_reviews": service.list_synthesis_reviews(tenant_id),
		"governance_events": service.list_governance_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def transcription_console_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"jobs": [job for job in service.list_jobs(tenant_id) if job["job_type"] == "transcription"],
		"pending_reviews": [review for review in service.list_transcript_reviews(tenant_id) if review["decision"] == "pending"],
		"required_fields": ["id", "audio_source_id", "requested_by", "model_id"],
		"review_required_below_confidence": 0.78,
	}


def synthesis_studio_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	synthesis_reviews = service.list_synthesis_reviews(tenant_id)
	return {
		"jobs": [job for job in service.list_jobs(tenant_id) if job["job_type"] in {"synthesis", "voice_cloning"}],
		"synthesis_reviews": synthesis_reviews,
		"pending_release_reviews": [review for review in synthesis_reviews if review["decision"] == "pending"],
		"required_controls": ["model_policy", "watermark_applied", "release_review", "voice_owner_consent_for_cloning"],
	}


def analysis_workbench_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"jobs": [job for job in service.list_jobs(tenant_id) if job["job_type"] == "analysis"],
		"analysis_types": ["sentiment", "topics", "quality", "speaker_characteristics", "content_classification"],
		"required_controls": ["recording_consent", "model_policy", "retention_policy"],
	}


def sessions_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"jobs": service.list_jobs(tenant_id),
		"summary": service.audio_summary(tenant_id),
	}


def model_registry_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"model_policies": service.list_model_policies(tenant_id),
		"allowed_operations": ["transcription", "synthesis", "analysis", "voice_cloning", "enhancement"],
		"required_fields": ["id", "model_id", "policy_name", "allowed_operations", "attached_by"],
	}


def consent_center_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"consents": service.list_consents(tenant_id),
		"consent_types": ["recording", "voice_owner"],
		"required_fields": ["id", "consent_type", "subject_id", "granted_by", "evidence"],
	}


def review_queue_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	return {
		"transcript_reviews": service.list_transcript_reviews(tenant_id),
		"synthesis_reviews": service.list_synthesis_reviews(tenant_id),
		"required_decision_fields": ["reviewer", "decision", "notes"],
	}


def quality_governance_model(
	service: AudpService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AudpService()
	contract = get_capability_contract(tenant_id)
	return {
		"summary": service.audio_summary(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"governance_events": service.list_governance_events(tenant_id),
		"theme": contract["theme"],
	}
