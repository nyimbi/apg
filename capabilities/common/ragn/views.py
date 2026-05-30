"""UI metadata helpers for APG Retrieval-Augmented Generation."""

from __future__ import annotations

from .capability_contract import get_capability_contract
from .rag_runtime import RagnService


def capability_routes(tenant_id: str = "default") -> list[dict[str, str]]:
	return list(get_capability_contract(tenant_id)["ui"]["routes"])


def dashboard_model(service: RagnService | None = None, tenant_id: str = "default") -> dict[str, object]:
	service = service or RagnService()
	contract = service.describe(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"routes": capability_routes(tenant_id),
		"summary": service.dashboard_summary(tenant_id),
		"knowledge_bases": service.list_knowledge_bases(tenant_id),
		"documents": service.list_documents(tenant_id),
		"retrievals": service.list_retrievals(tenant_id),
		"answers": service.list_answers(tenant_id),
		"conversations": service.list_conversations(tenant_id),
		"curations": service.list_curations(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"theme": contract["theme"],
	}


def studio_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"knowledge_bases": service.list_knowledge_bases(tenant_id),
		"retrievals": service.list_retrievals(tenant_id),
		"answers": service.list_answers(tenant_id),
	}


def knowledge_base_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"knowledge_bases": service.list_knowledge_bases(tenant_id),
		"documents": service.list_documents(tenant_id),
	}


def document_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"documents": service.list_documents(tenant_id),
		"restricted_documents": [
			document for document in service.list_documents(tenant_id)
			if document["metadata"].get("classification") == "restricted"
		],
	}


def retrieval_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"retrievals": service.list_retrievals(tenant_id),
		"documents": service.list_documents(tenant_id),
	}


def generation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"answers": service.list_answers(tenant_id),
		"retrievals": service.list_retrievals(tenant_id),
	}


def conversation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"conversation_turns": service.list_conversations(tenant_id),
		"answers": service.list_answers(tenant_id),
	}


def citation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	answers = service.list_answers(tenant_id)
	return {
		"tenant_id": tenant_id,
		"answers": answers,
		"citation_count": sum(int(answer["metadata"].get("citation_count", 0)) for answer in answers),
	}


def curation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"answers": service.list_answers(tenant_id),
		"curations": service.list_curations(tenant_id),
	}


def governance_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"audit_events": service.list_audit_events(tenant_id),
		"configuration": contract["configuration"],
	}


def audit_timeline_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	return {
		"tenant_id": tenant_id,
		"audit_events": service.list_audit_events(tenant_id),
	}


def settings_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"theme": contract["theme"],
		"adapters": contract["configuration"]["adapters"],
	}
