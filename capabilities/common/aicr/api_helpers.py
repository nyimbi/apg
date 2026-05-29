"""Dependency-light API helper surface for AICR package composition."""

from __future__ import annotations

from typing import Any

from .service import AicrService


SERVICE = AicrService()


def register_ai_service(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register a governed AI service from an API-shaped payload."""
	return SERVICE.register_ai_service(
		service_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		service_type=str(payload.get("service_type") or "inference"),
		endpoint=str(payload.get("endpoint") or "local://inference"),
		health=str(payload.get("health") or "healthy"),
		model_policy=dict(payload.get("model_policy") or {}),
	)


def request_inference(payload: dict[str, Any]) -> dict[str, Any]:
	"""Request governed inference from an API-shaped payload."""
	return SERVICE.request_inference(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		service_id=str(payload["service_id"]),
		requested_by=str(payload.get("requested_by") or ""),
		prompt_summary=str(payload.get("prompt_summary") or ""),
		model_policy_attached=bool(payload.get("model_policy_attached", True)),
		context_tokens=int(payload.get("context_tokens") or 0),
		workflow_risk=str(payload.get("workflow_risk") or "normal"),
	)


def decide_inference_approval(payload: dict[str, Any]) -> dict[str, Any]:
	"""Approve or reject governed inference from an API-shaped payload."""
	return SERVICE.decide_inference_approval(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		reviewer=str(payload["reviewer"]),
		decision=str(payload.get("decision") or "approved"),
		notes=str(payload.get("notes") or ""),
	)


def run_approved_inference(payload: dict[str, Any]) -> dict[str, Any]:
	"""Run an approved inference request from an API-shaped payload."""
	return SERVICE.run_approved_inference(
		request_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
	)


def list_ai_services(tenant_id: str | None = None) -> list[dict[str, Any]]:
	"""List registered AI services for the optional tenant."""
	return SERVICE.list_ai_services(tenant_id)


def list_inference_approvals(tenant_id: str | None = None) -> list[dict[str, Any]]:
	"""List inference approvals for the optional tenant."""
	return SERVICE.list_inference_approvals(tenant_id)


def list_audit_events(tenant_id: str | None = None) -> list[dict[str, Any]]:
	"""List AICR governance events for the optional tenant."""
	return SERVICE.list_audit_events(tenant_id)


__all__ = [
	"SERVICE",
	"register_ai_service",
	"request_inference",
	"decide_inference_approval",
	"run_approved_inference",
	"list_ai_services",
	"list_inference_approvals",
	"list_audit_events",
]
