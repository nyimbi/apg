"""Service layer for APG Knowledge Graph."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .knowledge_runtime import KnowledgeRuntime
from .models import (
	CurationRecord,
	GraphPublication,
	KngrAuditEvent,
	KnowledgeEntity,
	KnowledgeRelationship,
	KnowledgeSource,
	ReasoningPath,
	SemanticEnrichment,
)


class KngrService:
	"""Knowledge source, entity, enrichment, reasoning, curation, and publication service."""

	def __init__(self) -> None:
		self._sources: dict[str, KnowledgeSource] = {}
		self._entities: dict[str, KnowledgeEntity] = {}
		self._relationships: dict[str, KnowledgeRelationship] = {}
		self._enrichments: dict[str, SemanticEnrichment] = {}
		self._reasoning_paths: dict[str, ReasoningPath] = {}
		self._curations: dict[str, CurationRecord] = {}
		self._publications: dict[str, GraphPublication] = {}
		self._audit_events: dict[str, KngrAuditEvent] = {}
		self._runtime = KnowledgeRuntime()

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_source(
		self,
		source_id: str,
		tenant_id: str,
		name: str,
		source_uri: str,
		owner: str,
		evidence_refs: list[str] | tuple[str, ...],
		confidence_score: float,
		connector: str = "local",
		status: str = "active",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		if not owner:
			raise PermissionError("source_owner_required")
		if not source_uri:
			raise PermissionError("source_uri_required")
		if not evidence_refs:
			raise PermissionError("source_evidence_required")
		confidence = self._runtime.normalize_confidence(confidence_score)
		if confidence <= 0:
			raise PermissionError("source_confidence_required")
		source = KnowledgeSource(
			id=source_id,
			tenant_id=tenant_id,
			name=name,
			source_uri=source_uri,
			owner=owner,
			connector=connector,
			evidence_refs=tuple(evidence_refs),
			confidence_score=confidence,
			status=status,
		)
		self._sources[source_id] = source
		self._audit(tenant_id, source_id, "source_registered", owner, "allow", metadata={"connector": connector})
		return source.to_dict()

	def resolve_entity(
		self,
		entity_id: str,
		tenant_id: str,
		canonical_label: str,
		entity_type: str,
		source_id: str,
		source_evidence_refs: list[str] | tuple[str, ...],
		aliases: list[str] | tuple[str, ...] = (),
		attributes: dict[str, Any] | None = None,
		confidence_score: float = 1.0,
		curation_recorded: bool = False,
	) -> dict[str, Any]:
		source = self._require_source(source_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "resolve_entity",
			"source_evidence_present": bool(source_evidence_refs),
		})
		self._raise_if_denied(result)
		if not canonical_label:
			raise PermissionError("canonical_label_required")
		if not entity_type:
			raise PermissionError("entity_type_required")
		confidence = min(source.confidence_score, self._runtime.normalize_confidence(confidence_score))
		entity = KnowledgeEntity(
			id=entity_id,
			tenant_id=tenant_id,
			canonical_label=canonical_label,
			entity_type=entity_type,
			source_id=source_id,
			source_evidence_refs=tuple(source_evidence_refs),
			aliases=tuple(aliases),
			attributes=dict(attributes or {}),
			confidence_score=confidence,
			curation_status=self._runtime.entity_curation_status(curation_recorded, confidence),
		)
		self._entities[entity_id] = entity
		self._audit(
			tenant_id,
			entity_id,
			"entity_resolved",
			source.owner,
			result["decision"],
			reasons=self._reasons(result),
			metadata={"entity_type": entity_type, "source_id": source_id},
		)
		return entity.to_dict()

	def link_relationship(
		self,
		relationship_id: str,
		tenant_id: str,
		subject_entity_id: str,
		predicate: str,
		object_entity_id: str,
		source_id: str,
		evidence_links: list[str] | tuple[str, ...],
		confidence_score: float,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_entity(subject_entity_id, tenant_id)
		self._require_entity(object_entity_id, tenant_id)
		source = self._require_source(source_id, tenant_id)
		if not predicate:
			raise PermissionError("predicate_required")
		if not evidence_links:
			raise PermissionError("relationship_evidence_required")
		confidence = self._runtime.normalize_confidence(confidence_score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "enrich",
			"confidence_score": confidence,
		})
		self._raise_if_review_required(result, review_recorded)
		relationship = KnowledgeRelationship(
			id=relationship_id,
			tenant_id=tenant_id,
			subject_entity_id=subject_entity_id,
			predicate=predicate,
			object_entity_id=object_entity_id,
			source_id=source_id,
			evidence_links=tuple(evidence_links),
			confidence_score=confidence,
			status=self._runtime.relationship_status(confidence, review_recorded),
		)
		self._relationships[relationship_id] = relationship
		self._audit(
			tenant_id,
			relationship_id,
			"relationship_linked",
			source.owner,
			result["decision"],
			reasons=self._reasons(result),
			metadata={"predicate": predicate},
		)
		return relationship.to_dict()

	def enrich_entity(
		self,
		enrichment_id: str,
		tenant_id: str,
		entity_id: str,
		semantic_labels: list[str] | tuple[str, ...],
		attributes: dict[str, Any],
		evidence_links: list[str] | tuple[str, ...],
		confidence_score: float,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_entity(entity_id, tenant_id)
		if not semantic_labels:
			raise PermissionError("semantic_labels_required")
		if not evidence_links:
			raise PermissionError("enrichment_evidence_required")
		confidence = self._runtime.normalize_confidence(confidence_score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "enrich",
			"confidence_score": confidence,
		})
		self._raise_if_review_required(result, review_recorded)
		enrichment = SemanticEnrichment(
			id=enrichment_id,
			tenant_id=tenant_id,
			entity_id=entity_id,
			semantic_labels=tuple(semantic_labels),
			attributes=dict(attributes),
			evidence_links=tuple(evidence_links),
			confidence_score=confidence,
			review_recorded=review_recorded,
			status="accepted_with_review" if result["decision"] == "require_review" else "active",
		)
		self._enrichments[enrichment_id] = enrichment
		self._audit(tenant_id, enrichment_id, "entity_enriched", entity_id, result["decision"], reasons=self._reasons(result))
		return enrichment.to_dict()

	def build_reasoning_path(
		self,
		path_id: str,
		tenant_id: str,
		query: str,
		start_entity_id: str,
		end_entity_id: str,
		relationship_ids: list[str] | tuple[str, ...],
		evidence_links: list[str] | tuple[str, ...],
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_entity(start_entity_id, tenant_id)
		self._require_entity(end_entity_id, tenant_id)
		for relationship_id in relationship_ids:
			self._require_relationship(relationship_id, tenant_id)
		depth = self._runtime.path_depth(tuple(relationship_ids))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "reason",
			"evidence_links_present": bool(evidence_links),
			"reasoning_depth": depth,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		self._raise_if_review_required(result, review_recorded)
		path = ReasoningPath(
			id=path_id,
			tenant_id=tenant_id,
			query=query,
			start_entity_id=start_entity_id,
			end_entity_id=end_entity_id,
			relationship_ids=tuple(relationship_ids),
			evidence_links=tuple(evidence_links),
			reasoning_depth=depth,
			review_recorded=review_recorded,
			status="reviewed" if review_recorded else "active",
		)
		self._reasoning_paths[path_id] = path
		self._audit(tenant_id, path_id, "reasoning_path_built", start_entity_id, result["decision"], reasons=self._reasons(result))
		return path.to_dict()

	def curate_entity(
		self,
		curation_id: str,
		tenant_id: str,
		entity_id: str,
		curator: str,
		decision: str,
		evidence_links: list[str] | tuple[str, ...],
		notes: str = "",
	) -> dict[str, Any]:
		entity = self._require_entity(entity_id, tenant_id)
		if not curator:
			raise PermissionError("curator_required")
		if not evidence_links:
			raise PermissionError("curation_evidence_required")
		if decision not in {"approved", "rejected", "needs_revision"}:
			raise PermissionError("curation_decision_invalid")
		curation = CurationRecord(
			id=curation_id,
			tenant_id=tenant_id,
			entity_id=entity.id,
			curator=curator,
			decision=decision,
			evidence_links=tuple(evidence_links),
			notes=notes,
		)
		self._curations[curation_id] = curation
		if decision == "approved":
			self._entities[entity.id] = KnowledgeEntity(
				id=entity.id,
				tenant_id=entity.tenant_id,
				canonical_label=entity.canonical_label,
				entity_type=entity.entity_type,
				source_id=entity.source_id,
				source_evidence_refs=entity.source_evidence_refs,
				aliases=entity.aliases,
				attributes=entity.attributes,
				confidence_score=entity.confidence_score,
				curation_status="curated",
				status=entity.status,
				created_at=entity.created_at,
			)
		self._audit(tenant_id, curation_id, "entity_curated", curator, "allow", metadata={"entity_id": entity_id, "decision": decision})
		return curation.to_dict()

	def publish_graph(
		self,
		publication_id: str,
		tenant_id: str,
		name: str,
		entity_ids: list[str] | tuple[str, ...],
		relationship_ids: list[str] | tuple[str, ...],
		published_by: str,
		curation_recorded: bool,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_graph",
			"curation_recorded": bool(curation_recorded),
		})
		self._raise_if_denied(result)
		if not published_by:
			raise PermissionError("publisher_required")
		for entity_id in entity_ids:
			entity = self._require_entity(entity_id, tenant_id)
			if curation_recorded and entity.curation_status != "curated":
				raise PermissionError("entity_curation_required")
		for relationship_id in relationship_ids:
			self._require_relationship(relationship_id, tenant_id)
		publication = GraphPublication(
			id=publication_id,
			tenant_id=tenant_id,
			name=name,
			entity_ids=tuple(entity_ids),
			relationship_ids=tuple(relationship_ids),
			published_by=published_by,
			status=self._runtime.publication_status(len(entity_ids), len(relationship_ids)),
		)
		self._publications[publication_id] = publication
		self._audit(tenant_id, publication_id, "graph_published", published_by, result["decision"], metadata={"entity_count": len(entity_ids), "relationship_count": len(relationship_ids)})
		return publication.to_dict()

	def context_neighborhood(self, tenant_id: str, entity_id: str) -> dict[str, Any]:
		self._require_entity(entity_id, tenant_id)
		return self._runtime.neighborhood(entity_id, self.list_entities(tenant_id), self.list_relationships(tenant_id))

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		source_id = str(metadata.get("source_id") or f"source-{record_id}")
		if source_id not in self._sources:
			self.register_source(
				source_id=source_id,
				tenant_id=tenant_id,
				name=str(metadata.get("source_name") or "Manual source"),
				source_uri=str(metadata.get("source_uri") or f"manual://{record_id}"),
				owner=str(metadata.get("owner") or "system"),
				evidence_refs=tuple(metadata.get("evidence_refs") or (f"manual:{record_id}",)),
				confidence_score=float(metadata.get("confidence_score", 1.0)),
			)
		return self.resolve_entity(
			entity_id=record_id,
			tenant_id=tenant_id,
			canonical_label=str(metadata.get("canonical_label") or metadata.get("label") or record_id),
			entity_type=str(metadata.get("entity_type") or "record"),
			source_id=source_id,
			source_evidence_refs=tuple(metadata.get("evidence_refs") or (f"manual:{record_id}",)),
			aliases=tuple(metadata.get("aliases") or ()),
			attributes=metadata,
			confidence_score=float(metadata.get("confidence_score", 1.0)),
			curation_recorded=status == "curated",
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_entities(tenant_id)

	def list_sources(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._sources, tenant_id)

	def list_entities(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._entities, tenant_id)

	def list_relationships(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._relationships, tenant_id)

	def list_enrichments(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._enrichments, tenant_id)

	def list_reasoning_paths(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reasoning_paths, tenant_id)

	def list_curations(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._curations, tenant_id)

	def list_publications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._publications, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		entities = self.list_entities(tenant_id)
		relationships = self.list_relationships(tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": len(self.list_sources(tenant_id)),
			"entity_count": len(entities),
			"relationship_count": len(relationships),
			"enrichment_count": len(self.list_enrichments(tenant_id)),
			"reasoning_path_count": len(self.list_reasoning_paths(tenant_id)),
			"curation_count": len(self.list_curations(tenant_id)),
			"publication_count": len(self.list_publications(tenant_id)),
			"review_required_count": len([
				item for item in entities + relationships
				if item.get("status") == "review_required" or item.get("curation_status") == "review_required"
			]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_source(self, source_id: str, tenant_id: str) -> KnowledgeSource:
		source = self._sources.get(source_id)
		if source is None or source.tenant_id != tenant_id:
			raise KeyError("knowledge_source_not_found")
		return source

	def _require_entity(self, entity_id: str, tenant_id: str) -> KnowledgeEntity:
		entity = self._entities.get(entity_id)
		if entity is None or entity.tenant_id != tenant_id:
			raise KeyError("knowledge_entity_not_found")
		return entity

	def _require_relationship(self, relationship_id: str, tenant_id: str) -> KnowledgeRelationship:
		relationship = self._relationships.get(relationship_id)
		if relationship is None or relationship.tenant_id != tenant_id:
			raise KeyError("knowledge_relationship_not_found")
		return relationship

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(", ".join(self._reasons(result)) or "knowledge_policy_blocked")

	def _raise_if_review_required(self, result: dict[str, Any], review_recorded: bool) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review" and not review_recorded:
			raise PermissionError(", ".join(self._reasons(result)) or "knowledge_review_required")

	def _audit(
		self,
		tenant_id: str,
		subject_id: str,
		event_type: str,
		actor: str,
		decision: str,
		reasons: tuple[str, ...] = (),
		metadata: dict[str, Any] | None = None,
	) -> None:
		event_id = self._runtime.stable_id("audit", {
			"tenant_id": tenant_id,
			"subject_id": subject_id,
			"event_type": event_type,
			"actor": actor,
			"index": len(self._audit_events),
		})
		self._audit_events[event_id] = KngrAuditEvent(
			id=event_id,
			tenant_id=tenant_id,
			subject_id=subject_id,
			event_type=event_type,
			actor=actor,
			decision=decision,
			reasons=tuple(reason for reason in reasons if reason),
			metadata=dict(metadata or {}),
		)

	def _list(self, records: dict[str, Any], tenant_id: str | None = None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "knowledge_policy_blocked") for action in result.get("actions", ()))
