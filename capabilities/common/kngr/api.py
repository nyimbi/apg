"""API helpers for APG Knowledge Graph."""

from __future__ import annotations

from typing import Any

from .service import KngrService


SERVICE = KngrService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		**summary,
	}


def register_source(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_source(
		source_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		source_uri=str(payload["source_uri"]),
		owner=str(payload["owner"]),
		evidence_refs=tuple(payload.get("evidence_refs") or ()),
		confidence_score=float(payload.get("confidence_score", 1.0)),
		connector=str(payload.get("connector") or "local"),
		status=str(payload.get("status") or "active"),
	)


def resolve_entity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.resolve_entity(
		entity_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		canonical_label=str(payload["canonical_label"]),
		entity_type=str(payload["entity_type"]),
		source_id=str(payload["source_id"]),
		source_evidence_refs=tuple(payload.get("source_evidence_refs") or ()),
		aliases=tuple(payload.get("aliases") or ()),
		attributes=dict(payload.get("attributes") or {}),
		confidence_score=float(payload.get("confidence_score", 1.0)),
		curation_recorded=bool(payload.get("curation_recorded", False)),
	)


def link_relationship(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.link_relationship(
		relationship_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		subject_entity_id=str(payload["subject_entity_id"]),
		predicate=str(payload["predicate"]),
		object_entity_id=str(payload["object_entity_id"]),
		source_id=str(payload["source_id"]),
		evidence_links=tuple(payload.get("evidence_links") or ()),
		confidence_score=float(payload.get("confidence_score", 1.0)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def enrich_entity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.enrich_entity(
		enrichment_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		entity_id=str(payload["entity_id"]),
		semantic_labels=tuple(payload.get("semantic_labels") or ()),
		attributes=dict(payload.get("attributes") or {}),
		evidence_links=tuple(payload.get("evidence_links") or ()),
		confidence_score=float(payload.get("confidence_score", 1.0)),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def build_reasoning_path(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.build_reasoning_path(
		path_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		query=str(payload["query"]),
		start_entity_id=str(payload["start_entity_id"]),
		end_entity_id=str(payload["end_entity_id"]),
		relationship_ids=tuple(payload.get("relationship_ids") or ()),
		evidence_links=tuple(payload.get("evidence_links") or ()),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def curate_entity(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.curate_entity(
		curation_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		entity_id=str(payload["entity_id"]),
		curator=str(payload["curator"]),
		decision=str(payload["decision"]),
		evidence_links=tuple(payload.get("evidence_links") or ()),
		notes=str(payload.get("notes") or ""),
	)


def publish_graph(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_graph(
		publication_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload.get("name") or payload["id"]),
		entity_ids=tuple(payload.get("entity_ids") or ()),
		relationship_ids=tuple(payload.get("relationship_ids") or ()),
		published_by=str(payload["published_by"]),
		curation_recorded=bool(payload.get("curation_recorded", False)),
	)


def context_neighborhood(tenant_id: str, entity_id: str) -> dict[str, Any]:
	return SERVICE.context_neighborhood(tenant_id, entity_id)


def create_record(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_record(
		record_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		metadata=dict(payload.get("metadata") or {}),
		status=str(payload.get("status") or "active"),
	)


def list_records(tenant_id: str | None = None) -> list[dict[str, Any]]:
	return SERVICE.list_records(tenant_id)


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
