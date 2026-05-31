"""Dependency-light BIOP view models for generated APG applications."""

from __future__ import annotations

from typing import Any

from .biometric_runtime import BiopService


def dashboard_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"summary": service.biometric_summary(tenant_id),
		"consents": service.list_consents(tenant_id),
		"templates": service.list_templates(tenant_id),
		"verifications": service.list_verifications(tenant_id),
		"privacy_reviews": service.list_reviews(tenant_id, "privacy"),
		"match_reviews": service.list_reviews(tenant_id, "match"),
		"biometric_agents": service.list_biometric_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def consent_center_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	consents = service.list_consents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"consents": consents,
		"active_consents": [item for item in consents if item["status"] == "active"],
		"revoked_consents": [item for item in consents if item["status"] == "revoked"],
		"required_fields": ["subject_id", "purpose", "modalities", "jurisdictions", "granted_by", "evidence"],
	}


def template_vault_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	templates = service.list_templates(tenant_id)
	return {
		"tenant_id": tenant_id,
		"templates": templates,
		"active_templates": [item for item in templates if item["status"] == "active"],
		"retired_templates": [item for item in templates if item["status"] == "retired"],
		"required_enrollment_fields": ["subject_id", "modality", "template_hash", "encrypted", "quality_score", "consent_id", "retention_policy"],
		"guardrails": ["active_consent", "encrypted_template", "quality_threshold", "raw_sample_retention_disabled"],
	}


def verification_workbench_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	verifications = service.list_verifications(tenant_id)
	return {
		"tenant_id": tenant_id,
		"verifications": verifications,
		"verified": [item for item in verifications if item["status"] == "verified"],
		"pending_privacy_review": [item for item in verifications if item["status"] == "pending_privacy_review"],
		"pending_match_review": [item for item in verifications if item["status"] == "pending_match_review"],
		"rejected": [item for item in verifications if item["status"] == "rejected"],
		"required_request_fields": ["subject_id", "template_id", "modality", "match_confidence", "liveness_score"],
	}


def review_queue_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	privacy_reviews = service.list_reviews(tenant_id, "privacy")
	match_reviews = service.list_reviews(tenant_id, "match")
	return {
		"tenant_id": tenant_id,
		"privacy_reviews": privacy_reviews,
		"match_reviews": match_reviews,
		"pending_privacy_reviews": [item for item in privacy_reviews if item["status"] == "pending"],
		"pending_match_reviews": [item for item in match_reviews if item["status"] == "pending"],
		"decided_reviews": [item for item in [*privacy_reviews, *match_reviews] if item["status"] != "pending"],
		"required_decision_fields": ["reviewer", "decision", "notes"],
		"guardrails": ["independent_reviewer", "reviewer_notes_required", "stale_review_blocked"],
	}


def biometric_agent_roster_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	agents = service.list_biometric_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"component": "BiometricAgentRoster",
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"theme_component": "biometric_agent_roster",
	}


def lifecycle_batch_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"component": "BIOPLifecycleBatchMonitor",
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(
	service: BiopService | None = None,
	tenant_id: str = "default",
) -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"summary": service.biometric_summary(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def _service_or_default(service: BiopService | None) -> BiopService:
	if service is not None:
		return service
	try:
		from .api_helpers import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return BiopService()
