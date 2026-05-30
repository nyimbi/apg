"""UI metadata helpers for the Digital Forms and eSign capability."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .service import EsgnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(
	service: EsgnService | None = None,
	tenant_id: str = "default",
) -> dict[str, object]:
	service = service or EsgnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"templates": service.list_templates(tenant_id),
		"submissions": service.list_submissions(tenant_id),
		"envelopes": service.list_envelopes(tenant_id),
		"signing_ceremonies": service.list_ceremonies(tenant_id),
		"evidence_packages": service.list_evidence_packages(tenant_id),
		"signing_agents": service.list_signing_agents(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
	}


def form_library_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"templates": service.list_templates(tenant_id),
		"submissions": service.list_submissions(tenant_id),
		"states": ["draft", "pending_review", "published"],
	}


def envelope_console_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"envelopes": service.list_envelopes(tenant_id),
		"signing_ceremonies": service.list_ceremonies(tenant_id),
		"states": ["sent", "partially_signed", "completed", "review_required", "cancelled", "rejected"],
		"guardrails": ["document_hash", "expires_at", "tamper_seal", "routing_order", "recipient_consent"],
	}


def signing_room_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"envelopes": [item for item in service.list_envelopes(tenant_id) if item["status"] in {"sent", "partially_signed"}],
		"signing_ceremonies": service.list_ceremonies(tenant_id),
		"required_controls": ["identity_verified", "signature_intent", "routing_order_ready", "tamper_seal_valid"],
	}


def signing_agent_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"signing_agents": service.list_signing_agents(tenant_id),
		"supported_runtimes": contract["configuration"]["signing_agents"]["supported_runtimes"],
		"allowed_roles": contract["configuration"]["signing_agents"]["allowed_roles"],
		"required_controls": ["registered_by", "scope_ref", "contribution_disclosed"],
	}


def evidence_vault_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"evidence_packages": service.list_evidence_packages(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"required_controls": ["encrypted", "audit_trail_ref", "retention_policy", "certificate_id", "tamper_seal_valid"],
	}


def audit_trail_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
		"event_types": sorted({event["event_type"] for event in service.list_audit_events(tenant_id)}),
	}


def analytics_model(service: EsgnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or EsgnService()
	return {
		"tenant_id": tenant_id,
		"summary": service.dashboard_summary(tenant_id),
		"envelope_states": {state: len([item for item in service.list_envelopes(tenant_id) if item["status"] == state]) for state in ["sent", "partially_signed", "completed", "cancelled", "rejected"]},
	}
