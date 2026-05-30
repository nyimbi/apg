"""UI metadata helpers for the AI Core Framework capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import AicrService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"records": service.list_records(tenant_id),
		"providers": service.list_providers(tenant_id),
		"models": service.list_models(tenant_id),
		"workflows": service.list_workflows(tenant_id),
		"agent_runtimes": service.list_agent_runtimes(tenant_id),
		"summary": service.governance_summary(tenant_id),
		"inference_approvals": service.list_inference_approvals(tenant_id),
		"inference_results": service.list_inference_results(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def service_registry_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	return {
		"services": service.list_ai_services(tenant_id),
		"required_fields": ["id", "name", "owner", "model_policy", "health"],
		"health_states": ["healthy", "degraded", "unhealthy", "maintenance"],
	}


def provider_registry_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	contract = get_capability_contract(tenant_id)
	return {
		"providers": service.list_providers(tenant_id),
		"supported_provider_types": contract["configuration"]["providers"]["supported_provider_types"],
		"required_fields": ["id", "name", "provider_type", "owner", "credential_vault_ref", "egress_policy_ref"],
	}


def model_catalog_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	contract = get_capability_contract(tenant_id)
	return {
		"models": service.list_models(tenant_id),
		"providers": service.list_providers(tenant_id),
		"supported_modalities": contract["configuration"]["models"]["supported_modalities"],
	}


def inference_console_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	return {
		"services": service.list_ai_services(tenant_id),
		"pending_approvals": [
			approval for approval in service.list_inference_approvals(tenant_id)
			if approval["decision"] == "pending"
		],
		"results": service.list_inference_results(tenant_id),
		"request_fields": ["id", "service_id", "prompt_summary", "workflow_risk", "context_tokens"],
	}


def governance_center_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	return {
		"summary": service.governance_summary(tenant_id),
		"models": service.list_models(tenant_id),
		"workflows": service.list_workflows(tenant_id),
		"agent_runtimes": service.list_agent_runtimes(tenant_id),
		"approvals": service.list_inference_approvals(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
	}


def workflow_designer_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	return {
		"tenant_id": tenant_id,
		"workflows": service.list_workflows(tenant_id),
		"services": service.list_ai_services(tenant_id),
		"actions": ["create_workflow", "request_inference", "decide_inference_approval"],
	}


def agent_runtime_console_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	contract = get_capability_contract(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agent_runtimes": service.list_agent_runtimes(tenant_id),
		"supported_runtimes": contract["configuration"]["agent_runtimes"]["supported_runtimes"],
	}


def audit_timeline_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	return {
		"tenant_id": tenant_id,
		"events": service.list_audit_events(tenant_id),
		"observability": get_capability_contract(tenant_id)["configuration"]["observability"],
	}


def metrics_model(
	service: AicrService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or AicrService()
	summary = service.governance_summary(tenant_id)
	return {
		"tenant_id": tenant_id,
		"service_count": summary["service_count"],
		"provider_count": summary["provider_count"],
		"model_count": summary["model_count"],
		"workflow_count": summary["workflow_count"],
		"agent_runtime_count": summary["agent_runtime_count"],
		"healthy_service_count": summary["healthy_service_count"],
		"pending_approval_count": summary["pending_approval_count"],
		"inference_result_count": summary["inference_result_count"],
		"audit_event_count": summary["audit_event_count"],
	}
