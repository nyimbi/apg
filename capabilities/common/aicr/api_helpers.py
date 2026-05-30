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


def register_provider(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register an AI provider from an API-shaped payload."""
	return SERVICE.register_provider(
		provider_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		provider_type=str(payload.get("provider_type") or "local"),
		owner=str(payload.get("owner") or ""),
		external=bool(payload.get("external", True)),
		credential_vault_ref=str(payload.get("credential_vault_ref") or ""),
		egress_policy_ref=str(payload.get("egress_policy_ref") or ""),
	)


def register_model(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register an AI model from an API-shaped payload."""
	return SERVICE.register_model(
		model_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		provider_id=str(payload["provider_id"]),
		owner=str(payload.get("owner") or ""),
		modality=str(payload.get("modality") or "text"),
		model_policy=dict(payload.get("model_policy") or {}),
		risk_profile=str(payload.get("risk_profile") or "standard"),
	)


def create_workflow(payload: dict[str, Any]) -> dict[str, Any]:
	"""Create an AI workflow from an API-shaped payload."""
	return SERVICE.create_workflow(
		workflow_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		owner=str(payload.get("owner") or ""),
		service_ids=list(payload.get("service_ids") or []),
		risk=str(payload.get("risk") or "normal"),
	)


def register_agent_runtime(payload: dict[str, Any]) -> dict[str, Any]:
	"""Register an AI agent runtime from an API-shaped payload."""
	return SERVICE.register_agent_runtime(
		runtime_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		runtime_type=str(payload.get("runtime_type") or "codex"),
		owner=str(payload.get("owner") or ""),
		tool_policy_ref=str(payload.get("tool_policy_ref") or ""),
	)


def list_providers(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_providers(tenant_id)


def list_models(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_models(tenant_id)


def list_workflows(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_workflows(tenant_id)


def list_agent_runtimes(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_agent_runtimes(tenant_id)


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
	"register_provider",
	"register_model",
	"create_workflow",
	"register_agent_runtime",
	"list_providers",
	"list_models",
	"list_workflows",
	"list_agent_runtimes",
	"list_inference_approvals",
	"list_audit_events",
]
