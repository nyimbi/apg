"""API helpers for the Ontology Management capability."""

from __future__ import annotations

from typing import Any

from .service import OntoService


SERVICE = OntoService()


def capability_status(tenant_id: str = "default") -> dict[str, Any]:
	contract = SERVICE.describe(tenant_id)
	summary = SERVICE.dashboard_summary(tenant_id)
	return {
		"capability": contract["capability"],
		"display_name": contract["display_name"],
		"tenant_id": tenant_id,
		"route_count": len(contract["ui"]["routes"]),
		"rule_count": len(contract["rule_engine"]["rules"]),
		"ontology_count": summary["ontology_count"],
		"term_count": summary["term_count"],
		"mapping_count": summary["mapping_count"],
		"publication_count": summary["publication_count"],
	}


def register_ontology(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_ontology(
		ontology_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		name=str(payload["name"]),
		owner=str(payload.get("owner") or ""),
		domain=str(payload.get("domain") or "general"),
		description=str(payload.get("description") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def create_term(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_term(
		term_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		label=str(payload["label"]),
		owner=str(payload.get("owner") or ""),
		definition=str(payload.get("definition") or ""),
		status=str(payload.get("status") or "draft"),
		synonyms=list(payload.get("synonyms") or []),
		external_refs=list(payload.get("external_refs") or []),
		metadata=dict(payload.get("metadata") or {}),
		review_recorded=bool(payload.get("review_recorded", False)),
	)


def register_namespace(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.register_namespace(
		namespace_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		prefix=str(payload["prefix"]),
		uri=str(payload["uri"]),
		owner=str(payload.get("owner") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def curate_term(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.curate_term(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		term_id=str(payload["term_id"]),
		reviewer=str(payload.get("reviewer") or ""),
		status=str(payload.get("status") or "curated"),
		notes=str(payload.get("notes") or ""),
		change_type=str(payload.get("change_type") or "non_breaking"),
		review_recorded=bool(payload.get("review_recorded", True)),
	)


def add_synonym(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_synonym(
		tenant_id=str(payload.get("tenant_id") or "default"),
		term_id=str(payload["term_id"]),
		synonym=str(payload["synonym"]),
	)


def add_taxonomy_edge(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.add_taxonomy_edge(
		edge_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		parent_term_id=str(payload["parent_term_id"]),
		child_term_id=str(payload["child_term_id"]),
		relationship_type=str(payload.get("relationship_type") or "broader_than"),
	)


def deprecate_term(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.deprecate_term(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		term_id=str(payload["term_id"]),
		replacement_term_id=str(payload.get("replacement_term_id") or ""),
		reviewer=str(payload.get("reviewer") or ""),
		review_recorded=bool(payload.get("review_recorded", False)),
		notes=str(payload.get("notes") or ""),
	)


def create_mapping(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.create_mapping(
		mapping_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		term_id=str(payload["term_id"]),
		target_ref=str(payload["target_ref"]),
		mapping_type=str(payload.get("mapping_type") or "exact"),
		confidence=float(payload.get("confidence", 1.0)),
		review_recorded=bool(payload.get("review_recorded")),
		review_ref=str(payload.get("review_ref") or ""),
		metadata=dict(payload.get("metadata") or {}),
	)


def review_mapping(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.review_mapping(
		review_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		mapping_id=str(payload["mapping_id"]),
		reviewer=str(payload.get("reviewer") or ""),
		notes=str(payload.get("notes") or ""),
	)


def validate_ontology(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.validate_ontology(
		report_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		review_recorded=bool(payload.get("review_recorded", False)),
		review_ref=str(payload.get("review_ref") or ""),
	)


def publish_ontology(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.publish_ontology(
		publication_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		approval_recorded=bool(payload.get("approval_recorded")),
		approval_ref=str(payload.get("approval_ref") or ""),
	)


def export_ontology(payload: dict[str, Any]) -> dict[str, Any]:
	return SERVICE.export_ontology(
		export_id=str(payload["id"]),
		tenant_id=str(payload.get("tenant_id") or "default"),
		ontology_id=str(payload["ontology_id"]),
		export_format=str(payload.get("format") or "jsonld"),
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


def dashboard_summary(tenant_id: str = "default") -> dict[str, Any]:
	return SERVICE.dashboard_summary(tenant_id)
