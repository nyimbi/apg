"""Dependency-light FREC view models for generated APG applications."""

from __future__ import annotations

from typing import Any

from .face_runtime import FrecService


def dashboard_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {
		"component": "FRECDashboard",
		"summary": service.dashboard_summary(tenant_id),
		"routes": service.describe()["ui"]["routes"],
		"theme": service.describe()["theme"],
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


def audit_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FRECAuditTrail", "audit_events": service.list_audit_events(tenant_id), "theme_component": "audit_timeline"}


def settings_model(service: FrecService | None = None, tenant_id: str = "default") -> dict[str, Any]:
	service = _service_or_default(service)
	return {"component": "FRECSettings", "tenant_id": tenant_id, "configuration": service.describe()["configuration"]}


def _service_or_default(service: FrecService | None) -> FrecService:
	if service is not None:
		return service
	try:
		from .api_helpers import SERVICE

		return SERVICE
	except ImportError:  # pragma: no cover - standalone package loading path
		return FrecService()
