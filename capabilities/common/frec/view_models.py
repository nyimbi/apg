"""Dependency-light FREC view models for generated APG applications."""

from __future__ import annotations

from typing import Any

from .face_runtime import FrecService


def dashboard_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe()
	return {
		"component": "FRECDashboard",
		"summary": service.dashboard_summary(tenant_id),
		"facial_recognition_agents": service.list_facial_recognition_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"routes": contract["ui"]["routes"],
		"theme": contract["theme"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def consent_center_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FaceConsentCenter", "consents": service.list_consents(tenant_id), "theme_component": "consent_scope"}


def template_gallery_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FaceTemplateGallery", "templates": service.list_templates(tenant_id), "theme_component": "template_gallery"}


def verification_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FaceVerification", "verifications": service.list_verifications(tenant_id), "theme_component": "match_gallery"}


def identification_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FaceIdentification", "identifications": service.list_identifications(tenant_id), "watchlists": service.list_watchlists(tenant_id), "theme_component": "watchlist_table"}


def liveness_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FaceLiveness", "liveness": service.list_liveness(tenant_id), "theme_component": "liveness_trace"}


def review_queue_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	reviews = service.list_reviews(tenant_id)
	return {"component": "FaceReviewQueue", "reviews": reviews, "pending_reviews": [item for item in reviews if item["status"] == "pending"], "theme_component": "review_queue"}


def emotion_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "EmotionGovernance", "emotion_events": service.list_emotion_events(tenant_id), "theme_component": "emotion_governance"}


def facial_recognition_agent_roster_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe()
	agents = service.list_facial_recognition_agents(tenant_id)
	return {
		"component": "FacialRecognitionAgentRoster",
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
		"theme_component": "facial_recognition_agent_roster",
	}


def lifecycle_batch_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe()
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"component": "FRECLifecycleBatchMonitor",
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
		"theme_component": "bytewax_lifecycle_panel",
	}


def audit_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe()
	return {
		"component": "FRECAuditTrail",
		"audit_events": service.list_audit_events(tenant_id),
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme_component": "audit_timeline",
	}


def settings_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	contract = service.describe()
	return {
		"component": "FRECSettings",
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
	}


def _service_or_default(service: FrecService | None) -> FrecService:
	if service is not None:
		return service
	try:
		from .api_helpers import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return FrecService()
