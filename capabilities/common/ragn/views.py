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
		"rag_agents": service.list_rag_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
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
	documents = service.list_documents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"knowledge_bases": service.list_knowledge_bases(tenant_id),
		"documents": documents,
		"pending_review": [
			item for item in documents
			if item["status"] == "pending_review"
		],
	}


def document_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	documents = service.list_documents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"documents": documents,
		"restricted_documents": [
			document for document in documents
			if document["metadata"].get("classification") == "restricted"
		],
		"pending_review": [
			document for document in documents
			if document["status"] == "pending_review"
		],
	}


def retrieval_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	retrievals = service.list_retrievals(tenant_id)
	return {
		"tenant_id": tenant_id,
		"retrievals": retrievals,
		"documents": service.list_documents(tenant_id),
		"pending_review": [
			retrieval for retrieval in retrievals
			if retrieval["status"] == "pending_review"
		],
	}


def generation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	answers = service.list_answers(tenant_id)
	return {
		"tenant_id": tenant_id,
		"answers": answers,
		"retrievals": service.list_retrievals(tenant_id),
		"pending_review": [
			answer for answer in answers
			if answer["status"] == "pending_review"
		],
	}


def conversation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	turns = service.list_conversations(tenant_id)
	return {
		"tenant_id": tenant_id,
		"conversation_turns": turns,
		"answers": service.list_answers(tenant_id),
		"pending_review": [
			turn for turn in turns
			if turn["status"] == "pending_review"
		],
	}


def citation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	answers = service.list_answers(tenant_id)
	return {
		"tenant_id": tenant_id,
		"answers": answers,
		"citation_count": sum(int(answer["metadata"].get("citation_count", 0)) for answer in answers),
	}


def curation_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	answers = service.list_answers(tenant_id)
	return {
		"tenant_id": tenant_id,
		"answers": answers,
		"curations": service.list_curations(tenant_id),
		"pending_review": [
			answer for answer in answers
			if answer["status"] == "pending_review"
		],
	}


def governance_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	return {
		"tenant_id": tenant_id,
		"rules": contract["rule_engine"]["rules"],
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"rag_agents": service.list_rag_agents(tenant_id),
		"lifecycle_batches": service.list_lifecycle_batches(tenant_id),
		"pending_reviews": service.list_pending_reviews(tenant_id),
		"audit_events": service.list_audit_events(tenant_id),
		"configuration": contract["configuration"],
	}


def rag_agent_roster_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	agents = service.list_rag_agents(tenant_id)
	return {
		"tenant_id": tenant_id,
		"agents": agents,
		"pending_review": [item for item in agents if item["status"] == "pending_review"],
		"supported_runtimes": contract["agents"]["supported_runtimes"],
		"supported_roles": contract["agents"]["supported_roles"],
		"privileged_roles": contract["agents"]["privileged_roles"],
	}


def lifecycle_batch_model(service: RagnService, tenant_id: str = "default") -> dict[str, object]:
	contract = service.describe(tenant_id)
	batches = service.list_lifecycle_batches(tenant_id)
	return {
		"tenant_id": tenant_id,
		"batches": batches,
		"denied": [item for item in batches if item["status"] == "denied"],
		"required_processor": contract["streaming"]["required_processor"],
		"required_operations": contract["streaming"]["required_operations"],
		"topics": contract["streaming"]["topics"],
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
		"agents": contract["agents"],
		"streaming": contract["streaming"],
		"theme": contract["theme"],
		"adapters": contract["configuration"]["adapters"],
	}
