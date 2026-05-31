"""API helpers for the Search Engine capability."""

from __future__ import annotations

from typing import Any

from .service import SrchService


SERVICE = SrchService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"index_count": summary["index_count"],
		"document_count": summary["document_count"],
		"query_count": summary["query_count"],
		"review_required_query_count": summary["review_required_query_count"],
		"search_agent_count": summary["search_agent_count"],
		"lifecycle_batch_count": summary["lifecycle_batch_count"],
	}


def create_index(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_index(
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		content_type=str(payload.get("content_type") or ""),
		classification=str(payload.get("classification") or ""),
		source_lineage_ref=payload.get("source_lineage_ref"),
		embedding_index_ready=bool(payload.get("embedding_index_ready", False)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def mark_embedding_index_ready(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.mark_embedding_index_ready(
		tenant_id=str(payload.get("tenant_id") or "default"),
		index_id=str(payload["index_id"]),
		actor=str(payload.get("actor") or "api"),
	)


def index_document(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.index_document(
		tenant_id=str(payload.get("tenant_id") or "default"),
		index_id=str(payload["index_id"]),
		document_id=str(payload["document_id"]),
		title=str(payload.get("title") or ""),
		body=str(payload.get("body") or ""),
		classification=payload.get("classification"),
		facets=dict(payload.get("facets") or {}),
		metadata=dict(payload.get("metadata") or {}),
		source_lineage_ref=payload.get("source_lineage_ref"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def bulk_index_documents(payload: dict[str, Any]) -> list[dict[str, Any]]:
	return SERVICE.bulk_index_documents(
		tenant_id=str(payload.get("tenant_id") or "default"),
		index_id=str(payload["index_id"]),
		documents=list(payload.get("documents") or []),
		source_lineage_ref=payload.get("source_lineage_ref"),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def query(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.query(
		tenant_id=str(payload.get("tenant_id") or "default"),
		query_text=str(payload["query_text"]),
		index_ids=list(payload.get("index_ids") or []),
		query_type=str(payload.get("query_type") or ""),
		result_window=int(payload.get("result_window", 10)),
		rbac_filter_applied=bool(payload.get("rbac_filter_applied", True)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def register_search_agent(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_search_agent(
		agent_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or ""),
		runtime=str(payload.get("runtime") or ""),
		role=str(payload.get("role") or ""),
		scope=str(payload.get("scope") or ""),
		owner=str(payload.get("owner") or ""),
		purpose=str(payload.get("purpose") or ""),
		contribution_disclosed=bool(payload.get("contribution_disclosed", True)),
		human_approval_required=bool(payload.get("human_approval_required", False)),
	)


def validate_srch_lifecycle_batch(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_srch_lifecycle_batch(
		tenant_id=str(payload.get("tenant_id") or "default"),
		event_stream=str(payload.get("event_stream") or ""),
		mutation_count=int(payload.get("mutation_count") or 0),
		operation=str(payload.get("operation") or "search_agent_batch"),
		batch_id=payload.get("id"),
	)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def list_search_engine(tenant_id: str = "default") -> dict[str, Any]:
	return {
		"indices": SERVICE.list_indices(tenant_id),
		"documents": SERVICE.list_documents(tenant_id),
		"queries": SERVICE.list_queries(tenant_id),
		"search_agents": SERVICE.list_search_agents(tenant_id),
		"lifecycle_batches": SERVICE.list_lifecycle_batches(tenant_id),
		"audit_events": SERVICE.list_audit_events(tenant_id),
		"facets": SERVICE.facets(tenant_id),
		"summary": SERVICE.dashboard_summary(tenant_id),
	}


def list_search_agents(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_search_agents(tenant_id)


def list_lifecycle_batches(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_lifecycle_batches(tenant_id)
