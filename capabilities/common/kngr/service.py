"""Service layer for APG Knowledge Graph."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .knowledge_runtime import KnowledgeRuntime
from .models import (
	CurationRecord,
	GraphPublication,
	KngrAuditEvent,
	KngrLifecycleBatchRecord,
	KnowledgeAgentRecord,
	KnowledgeEntity,
	KnowledgeRelationship,
	KnowledgeSource,
	ReasoningPath,
	SemanticEnrichment,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
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
		self._knowledge_agents: dict[str, KnowledgeAgentRecord] = {}
		self._lifecycle_batches: dict[str, KngrLifecycleBatchRecord] = {}
		self._audit_events: dict[str, KngrAuditEvent] = {}
		self._runtime = KnowledgeRuntime()
		contract = get_capability_contract()
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		confidence = self._runtime.normalize_confidence(confidence_score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_source",
			"source_id_present": bool(source_id),
			"source_name_present": bool(name),
			"source_uri_present": bool(source_uri),
			"source_owner_present": bool(owner),
			"source_evidence_present": bool(evidence_refs),
			"confidence_score": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		record_status = "pending_review" if result["decision"] == "require_review" else status
		source = KnowledgeSource(
			id=source_id,
			tenant_id=tenant_id,
			name=name,
			source_uri=source_uri,
			owner=owner,
			connector=connector,
			evidence_refs=tuple(evidence_refs),
			confidence_score=confidence,
			status=record_status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
		)
		self._sources[source_id] = source
		self._audit(tenant_id, source_id, "source_registered", owner, result["decision"], reasons=self._reasons(result), metadata={"connector": connector})
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
		review_recorded: bool = False,
	) -> dict[str, Any]:
		source = self._require_source(source_id, tenant_id)
		confidence = min(source.confidence_score, self._runtime.normalize_confidence(confidence_score))
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "resolve_entity",
			"entity_id_present": bool(entity_id),
			"canonical_label_present": bool(canonical_label),
			"entity_type_present": bool(entity_type),
			"source_present": bool(source_id),
			"source_evidence_present": bool(source_evidence_refs),
			"confidence_score": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		record_status = "pending_review" if result["decision"] == "require_review" else "active"
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
			status=record_status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
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
		confidence = self._runtime.normalize_confidence(confidence_score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "link_relationship",
			"subject_present": bool(subject_entity_id),
			"object_present": bool(object_entity_id),
			"predicate_present": bool(predicate),
			"source_present": bool(source_id),
			"evidence_links_present": bool(evidence_links),
			"confidence_score": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		record_status = "pending_review" if result["decision"] == "require_review" else self._runtime.relationship_status(confidence, review_recorded)
		relationship = KnowledgeRelationship(
			id=relationship_id,
			tenant_id=tenant_id,
			subject_entity_id=subject_entity_id,
			predicate=predicate,
			object_entity_id=object_entity_id,
			source_id=source_id,
			evidence_links=tuple(evidence_links),
			confidence_score=confidence,
			status=record_status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
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
		confidence = self._runtime.normalize_confidence(confidence_score)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "enrich",
			"semantic_labels_present": bool(semantic_labels),
			"evidence_links_present": bool(evidence_links),
			"confidence_score": confidence,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else (
			"accepted_with_review" if confidence < 0.7 and review_recorded else "active"
		)
		enrichment = SemanticEnrichment(
			id=enrichment_id,
			tenant_id=tenant_id,
			entity_id=entity_id,
			semantic_labels=tuple(semantic_labels),
			attributes=dict(attributes),
			evidence_links=tuple(evidence_links),
			confidence_score=confidence,
			review_recorded=review_recorded,
			status=status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
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
			"query_present": bool(query),
			"entity_endpoints_present": bool(start_entity_id and end_entity_id),
			"evidence_links_present": bool(evidence_links),
			"reasoning_depth": depth,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		status = "pending_review" if result["decision"] == "require_review" else ("reviewed" if review_recorded else "active")
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
			status=status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
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
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "curate_entity",
			"curator_present": bool(curator),
			"curation_decision_present": bool(decision),
			"curation_decision_allowed": decision in {"approved", "rejected", "needs_revision"},
			"evidence_links_present": bool(evidence_links),
		})
		self._raise_if_denied(result)
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
				status="accepted_with_review" if entity.status == "pending_review" else entity.status,
				decision=entity.decision,
				matched_rules=entity.matched_rules,
				review_reasons=entity.review_reasons,
				created_at=entity.created_at,
			)
		self._audit(tenant_id, curation_id, "entity_curated", curator, result["decision"], reasons=self._reasons(result), metadata={"entity_id": entity_id, "decision": decision})
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
			"publication_name_present": bool(name),
			"publisher_present": bool(published_by),
			"curation_recorded": bool(curation_recorded),
			"entity_count": len(entity_ids),
		})
		self._raise_if_denied(result)
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

	def register_knowledge_agent(
		self,
		agent_id: str,
		tenant_id: str,
		name: str,
		runtime: str,
		role: str,
		scope: str,
		owner: str,
		purpose: str,
		contribution_disclosed: bool = True,
		human_approval_required: bool = False,
	) -> dict[str, Any]:
		runtime_value = _normalize_token(runtime)
		role_value = _normalize_token(role)
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "register_knowledge_agent",
			"agent_runtime_supported": runtime_value in self._agent_runtimes,
			"agent_role_supported": role_value in self._agent_roles,
			"scope_present": bool(str(scope or "").strip()),
			"owner_present": bool(str(owner or "").strip()),
			"purpose_present": bool(str(purpose or "").strip()),
			"contribution_disclosed": bool(contribution_disclosed),
			"privileged_role": role_value in self._privileged_agent_roles,
			"human_approval_required": bool(human_approval_required),
		})
		self._raise_if_denied(result)
		if not str(agent_id or "").strip():
			raise ValueError("knowledge_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("knowledge_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = KnowledgeAgentRecord(
			id=str(agent_id).strip(),
			tenant_id=tenant_id,
			name=str(name).strip(),
			runtime=runtime_value,
			role=role_value,
			scope=str(scope).strip(),
			owner=str(owner).strip(),
			purpose=str(purpose).strip(),
			contribution_disclosed=bool(contribution_disclosed),
			human_approval_required=bool(human_approval_required),
			status=status,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
		)
		self._knowledge_agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._audit(
			tenant_id,
			record.id,
			"knowledge_agent_registered",
			owner,
			result["decision"],
			reasons=self._reasons(result),
			metadata={"runtime": runtime_value, "role": role_value, "status": status},
		)
		return record.to_dict()

	def validate_kngr_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "knowledge_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("kngr_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_kngr_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_kngr_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = KngrLifecycleBatchRecord(
			id=batch_id or f"kngrbatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=tuple(result["matched_rules"]),
			review_reasons=self._review_reasons(result),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, record.id)] = record
		self._audit(
			tenant_id,
			record.id,
			f"kngr_lifecycle_batch_{record.status}",
			"kngr",
			result["decision"],
			reasons=self._reasons(result),
			metadata={"operation": operation_value, "event_stream": stream_value},
		)
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_entities(tenant_id)

	def list_knowledge_graph(self, tenant_id: str | None = None) -> dict[str, Any]:
		sources = self.list_sources(tenant_id)
		entities = self.list_entities(tenant_id)
		relationships = self.list_relationships(tenant_id)
		enrichments = self.list_enrichments(tenant_id)
		reasoning_paths = self.list_reasoning_paths(tenant_id)
		curations = self.list_curations(tenant_id)
		publications = self.list_publications(tenant_id)
		knowledge_agents = self.list_knowledge_agents(tenant_id)
		lifecycle_batches = self.list_lifecycle_batches(tenant_id)
		audit_events = self.list_audit_events(tenant_id)
		return {
			"tenant_id": tenant_id,
			"sources": sources,
			"entities": entities,
			"relationships": relationships,
			"enrichments": enrichments,
			"reasoning_paths": reasoning_paths,
			"curations": curations,
			"publications": publications,
			"knowledge_agents": knowledge_agents,
			"lifecycle_batches": lifecycle_batches,
			"audit_events": audit_events,
			"summary": {
				"tenant_id": tenant_id,
				"source_count": len(sources),
				"entity_count": len(entities),
				"relationship_count": len(relationships),
				"enrichment_count": len(enrichments),
				"reasoning_path_count": len(reasoning_paths),
				"curation_count": len(curations),
				"publication_count": len(publications),
				"knowledge_agent_count": len(knowledge_agents),
				"pending_source_review_count": len(self._pending_review(sources)),
				"pending_entity_review_count": len(self._pending_review(entities)),
				"pending_relationship_review_count": len(self._pending_review(relationships)),
				"pending_enrichment_review_count": len(self._pending_review(enrichments)),
				"pending_reasoning_review_count": len(self._pending_review(reasoning_paths)),
				"pending_agent_review_count": len([item for item in knowledge_agents if item["status"] == "pending_review"]),
				"lifecycle_batch_count": len(lifecycle_batches),
				"denied_lifecycle_batch_count": len([item for item in lifecycle_batches if item["status"] == "denied"]),
				"review_required_count": len([
					item for item in sources + entities + relationships + enrichments + reasoning_paths
					if item.get("status") in {"pending_review", "review_required"} or item.get("curation_status") == "review_required"
				]),
				"audit_event_count": len(audit_events),
			},
		}

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

	def list_knowledge_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._knowledge_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		sources = self.list_sources(tenant_id)
		entities = self.list_entities(tenant_id)
		relationships = self.list_relationships(tenant_id)
		enrichments = self.list_enrichments(tenant_id)
		reasoning_paths = self.list_reasoning_paths(tenant_id)
		knowledge_agents = self.list_knowledge_agents(tenant_id)
		lifecycle_batches = self.list_lifecycle_batches(tenant_id)
		return {
			"tenant_id": tenant_id,
			"source_count": len(sources),
			"entity_count": len(entities),
			"relationship_count": len(relationships),
			"enrichment_count": len(enrichments),
			"reasoning_path_count": len(reasoning_paths),
			"curation_count": len(self.list_curations(tenant_id)),
			"publication_count": len(self.list_publications(tenant_id)),
			"knowledge_agent_count": len(knowledge_agents),
			"pending_source_review_count": len(self._pending_review(sources)),
			"pending_entity_review_count": len(self._pending_review(entities)),
			"pending_relationship_review_count": len(self._pending_review(relationships)),
			"pending_enrichment_review_count": len(self._pending_review(enrichments)),
			"pending_reasoning_review_count": len(self._pending_review(reasoning_paths)),
			"pending_agent_review_count": len([item for item in knowledge_agents if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(lifecycle_batches),
			"denied_lifecycle_batch_count": len([item for item in lifecycle_batches if item["status"] == "denied"]),
			"review_required_count": len([
				item for item in sources + entities + relationships + enrichments + reasoning_paths
				if item.get("status") in {"pending_review", "review_required"} or item.get("curation_status") == "review_required"
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

	# ── 14 new methods ──────────────────────────────────────────────────────────

	def entity_update(
		self,
		entity_id: str,
		tenant_id: str,
		properties: dict[str, Any],
		actor: str = "system",
	) -> dict[str, Any]:
		"""Update mutable properties on an existing entity."""
		entity = self._require_entity(entity_id, tenant_id)
		for k, v in properties.items():
			if hasattr(entity, k):
				setattr(entity, k, v)
		self._audit(tenant_id, entity_id, "entity_updated", actor, "allow",
			metadata={"updated_keys": list(properties.keys())})
		return entity.to_dict()

	def entity_delete(
		self,
		entity_id: str,
		tenant_id: str,
		cascade: bool = False,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Delete an entity, optionally cascading to its relationships."""
		entity = self._require_entity(entity_id, tenant_id)
		deleted_rels: list[str] = []
		if cascade:
			for rel_id, rel in list(self._relationships.items()):
				if rel.tenant_id == tenant_id and (
					rel.subject_entity_id == entity_id or rel.object_entity_id == entity_id
				):
					del self._relationships[rel_id]
					deleted_rels.append(rel_id)
		del self._entities[entity_id]
		self._audit(tenant_id, entity_id, "entity_deleted", actor, "allow",
			metadata={"cascade": cascade, "deleted_relationships": deleted_rels})
		return {"entity_id": entity_id, "deleted": True, "cascade": cascade, "deleted_relationships": deleted_rels}

	def subgraph_extract(
		self,
		root_id: str,
		tenant_id: str,
		depth: int = 2,
	) -> dict[str, Any]:
		"""Extract a subgraph rooted at root_id up to a given hop depth."""
		visited: set[str] = set()
		frontier = {root_id}
		edges: list[dict[str, Any]] = []
		for _ in range(depth):
			next_frontier: set[str] = set()
			for rel in self._relationships.values():
				if rel.tenant_id != tenant_id:
					continue
				if rel.subject_entity_id in frontier and rel.object_entity_id not in visited:
					edges.append(rel.to_dict())
					next_frontier.add(rel.object_entity_id)
			visited |= frontier
			frontier = next_frontier - visited
		all_nodes = visited | frontier
		nodes = [e.to_dict() for eid, e in self._entities.items() if eid in all_nodes and e.tenant_id == tenant_id]
		return {
			"root_id": root_id,
			"depth": depth,
			"node_count": len(nodes),
			"edge_count": len(edges),
			"nodes": nodes,
			"edges": edges,
		}

	def path_find(
		self,
		from_id: str,
		to_id: str,
		tenant_id: str,
		max_hops: int = 5,
	) -> dict[str, Any]:
		"""BFS shortest path between two entities in the knowledge graph."""
		from collections import deque
		parent: dict[str, str | None] = {from_id: None}
		queue: deque[str] = deque([from_id])
		found = False
		hops = 0
		while queue and hops < max_hops:
			node = queue.popleft()
			if node == to_id:
				found = True
				break
			for rel in self._relationships.values():
				if rel.tenant_id != tenant_id or rel.subject_entity_id != node:
					continue
				nxt = rel.object_entity_id
				if nxt not in parent:
					parent[nxt] = node
					queue.append(nxt)
			hops += 1
		path: list[str] = []
		if found:
			cur: str | None = to_id
			while cur is not None:
				path.append(cur)
				cur = parent.get(cur)
			path.reverse()
		return {"from_id": from_id, "to_id": to_id, "path": path, "found": found, "hops": len(path) - 1}

	def community_detect(
		self,
		tenant_id: str,
		algorithm: str = "louvain",
	) -> dict[str, Any]:
		"""Detect communities using a simple connected-components proxy for Louvain."""
		entities = {e.id for e in self._entities.values() if e.tenant_id == tenant_id}
		adjacency: dict[str, set[str]] = {eid: set() for eid in entities}
		for rel in self._relationships.values():
			if rel.tenant_id != tenant_id:
				continue
			if rel.subject_entity_id in adjacency:
				adjacency[rel.subject_entity_id].add(rel.object_entity_id)
			if rel.object_entity_id in adjacency:
				adjacency[rel.object_entity_id].add(rel.subject_entity_id)
		visited: set[str] = set()
		communities: list[list[str]] = []
		for eid in entities:
			if eid in visited:
				continue
			stack = [eid]
			component: list[str] = []
			while stack:
				node = stack.pop()
				if node in visited:
					continue
				visited.add(node)
				component.append(node)
				stack.extend(adjacency.get(node, set()) - visited)
			communities.append(component)
		return {
			"algorithm": algorithm,
			"tenant_id": tenant_id,
			"community_count": len(communities),
			"communities": [{"id": i, "members": c} for i, c in enumerate(communities)],
		}

	def centrality_compute(
		self,
		tenant_id: str,
		metric: str = "pagerank",
	) -> dict[str, Any]:
		"""Compute degree-based centrality (proxy for PageRank) for all entities."""
		in_degree: dict[str, int] = {}
		out_degree: dict[str, int] = {}
		for rel in self._relationships.values():
			if rel.tenant_id != tenant_id:
				continue
			out_degree[rel.subject_entity_id] = out_degree.get(rel.subject_entity_id, 0) + 1
			in_degree[rel.object_entity_id] = in_degree.get(rel.object_entity_id, 0) + 1
		all_ids = set(in_degree) | set(out_degree)
		scores = {
			eid: round((in_degree.get(eid, 0) + out_degree.get(eid, 0)) / max(len(all_ids), 1), 4)
			for eid in all_ids
		}
		ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
		return {
			"metric": metric,
			"tenant_id": tenant_id,
			"entity_count": len(all_ids),
			"top_10": [{"entity_id": eid, "score": s} for eid, s in ranked[:10]],
		}

	def graph_merge(
		self,
		source_id: str,
		target_id: str,
		tenant_id: str,
		merge_strategy: str = "keep_target",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Merge source entity into target, re-pointing all relationships."""
		source = self._require_entity(source_id, tenant_id)
		target = self._require_entity(target_id, tenant_id)
		redirected: list[str] = []
		for rel in self._relationships.values():
			if rel.tenant_id != tenant_id:
				continue
			if rel.subject_entity_id == source_id:
				rel.subject_entity_id = target_id
				redirected.append(rel.id)
			if rel.object_entity_id == source_id:
				rel.object_entity_id = target_id
				if rel.id not in redirected:
					redirected.append(rel.id)
		del self._entities[source_id]
		self._audit(tenant_id, target_id, "graph_merge", actor, "allow",
			metadata={"source_id": source_id, "strategy": merge_strategy, "redirected": len(redirected)})
		return {
			"source_id": source_id,
			"target_id": target_id,
			"merge_strategy": merge_strategy,
			"relationships_redirected": len(redirected),
			"merged_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	def import_triples(
		self,
		triples_list: list[dict[str, Any]],
		tenant_id: str,
		source_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Bulk import RDF-style triples as entities + relationships.

		Each triple: {"subject": str, "predicate": str, "object": str}
		"""
		imported_entities: list[str] = []
		imported_rels: list[str] = []
		for triple in triples_list:
			s = triple["subject"]
			p = triple["predicate"]
			o = triple["object"]
			for eid, label in [(s, s), (o, o)]:
				if eid not in self._entities:
					self.resolve_entity(
						entity_id=eid, tenant_id=tenant_id, canonical_label=label,
						entity_type="concept", source_id=source_id, source_evidence_refs=[],
					)
					imported_entities.append(eid)
			rel_id = f"triple-{s}-{p}-{o}"
			if rel_id not in self._relationships:
				self.link_relationship(
					relationship_id=rel_id, tenant_id=tenant_id, subject_entity_id=s,
					predicate=p, object_entity_id=o, source_id=source_id, evidence_links=[],
				)
				imported_rels.append(rel_id)
		return {
			"tenant_id": tenant_id,
			"triples_submitted": len(triples_list),
			"entities_created": len(imported_entities),
			"relationships_created": len(imported_rels),
		}

	def export_jsonld(
		self,
		tenant_id: str,
		entity_ids: list[str] | None = None,
	) -> dict[str, Any]:
		"""Export entities and relationships as a JSON-LD document."""
		entities = [
			e for e in self._entities.values()
			if e.tenant_id == tenant_id and (entity_ids is None or e.id in entity_ids)
		]
		rels = [
			r for r in self._relationships.values()
			if r.tenant_id == tenant_id
			and (entity_ids is None or r.subject_entity_id in entity_ids or r.object_entity_id in entity_ids)
		]
		return {
			"@context": "https://schema.org/",
			"@graph": [
				{**e.to_dict(), "@type": e.entity_type} for e in entities
			],
			"relationships": [r.to_dict() for r in rels],
			"entity_count": len(entities),
			"relationship_count": len(rels),
		}

	def fact_validate(
		self,
		subject: str,
		predicate: str,
		object: str,
		tenant_id: str,
	) -> dict[str, Any]:
		"""Check whether a specific (subject, predicate, object) triple exists in the graph."""
		for rel in self._relationships.values():
			if (rel.tenant_id == tenant_id and rel.subject_entity_id == subject
					and rel.predicate == predicate and rel.object_entity_id == object):
				return {"valid": True, "relationship_id": rel.id, "confidence": rel.confidence_score}
		return {"valid": False, "subject": subject, "predicate": predicate, "object": object}

	def provenance_record(
		self,
		fact_id: str,
		source: str,
		confidence: float,
		tenant_id: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Record provenance metadata for a fact (relationship)."""
		prov_id = self._runtime.stable_id("prov", {
			"tenant_id": tenant_id, "fact_id": fact_id, "source": source,
		})
		self._audit(tenant_id, fact_id, "provenance_recorded", actor, "allow",
			metadata={"source": source, "confidence": confidence, "prov_id": prov_id})
		return {
			"provenance_id": prov_id,
			"fact_id": fact_id,
			"source": source,
			"confidence": confidence,
			"recorded_at": __import__("datetime").datetime.utcnow().isoformat(),
		}

	def kg_analytics(
		self,
		period: str,
		tenant_id: str = "default",
	) -> dict[str, Any]:
		"""Return knowledge graph analytics for a period."""
		entities = self.list_entities(tenant_id)
		rels = self.list_relationships(tenant_id)
		enrichments = self.list_enrichments(tenant_id)
		paths = self.list_reasoning_paths(tenant_id)
		return {
			"tenant_id": tenant_id,
			"period": period,
			"entity_count": len(entities),
			"relationship_count": len(rels),
			"enrichment_count": len(enrichments),
			"reasoning_path_count": len(paths),
			"avg_confidence": round(
				sum(r.get("confidence_score", 0) for r in rels) / max(len(rels), 1), 3
			),
			"audit_events": len(self.list_audit_events(tenant_id)),
		}

	def _pending_review(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
		return [record for record in records if record.get("status") == "pending_review"]

	def _reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(action.get("reason", "knowledge_policy_blocked") for action in result.get("actions", ()))

	def _review_reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		return tuple(
			action.get("reason", "knowledge_review_required")
			for action in result.get("actions", ())
			if action.get("decision") == "require_review"
		)

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


# ── Extended methods injected onto KngrService ────────────────────────────────

def _entity_add(
	self: "KngrService",
	entity_id: str,
	tenant_id: str,
	canonical_label: str,
	entity_type: str,
	source_id: str,
	evidence_refs: list[str],
	confidence_score: float = 1.0,
	aliases: list[str] | None = None,
	attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
	"""Spec alias for resolve_entity."""
	return self.resolve_entity(
		entity_id=entity_id,
		tenant_id=tenant_id,
		canonical_label=canonical_label,
		entity_type=entity_type,
		source_id=source_id,
		source_evidence_refs=evidence_refs,
		aliases=list(aliases or []),
		attributes=attributes,
		confidence_score=confidence_score,
	)

KngrService.entity_add = _entity_add  # type: ignore[attr-defined]


def _relation_add(
	self: "KngrService",
	relationship_id: str,
	tenant_id: str,
	subject_entity_id: str,
	predicate: str,
	object_entity_id: str,
	source_id: str,
	evidence_links: list[str],
	confidence_score: float = 1.0,
) -> dict[str, Any]:
	"""Spec alias for link_relationship."""
	return self.link_relationship(
		relationship_id=relationship_id,
		tenant_id=tenant_id,
		subject_entity_id=subject_entity_id,
		predicate=predicate,
		object_entity_id=object_entity_id,
		source_id=source_id,
		evidence_links=evidence_links,
		confidence_score=confidence_score,
	)

KngrService.relation_add = _relation_add  # type: ignore[attr-defined]


def _entity_merge(
	self: "KngrService",
	tenant_id: str,
	primary_entity_id: str,
	secondary_entity_id: str,
	merged_by: str,
) -> dict[str, Any]:
	"""Merge secondary entity into primary: copy aliases, mark secondary retired."""
	primary = self._require_entity(primary_entity_id, tenant_id)
	secondary = self._require_entity(secondary_entity_id, tenant_id)
	merged_aliases = tuple(set(primary.aliases) | set(secondary.aliases) | (secondary.canonical_label,))
	from .models import KnowledgeEntity
	self._entities[primary_entity_id] = KnowledgeEntity(
		id=primary.id,
		tenant_id=primary.tenant_id,
		canonical_label=primary.canonical_label,
		entity_type=primary.entity_type,
		source_id=primary.source_id,
		source_evidence_refs=primary.source_evidence_refs,
		aliases=merged_aliases,
		attributes={**secondary.attributes, **primary.attributes},
		confidence_score=primary.confidence_score,
		curation_status=primary.curation_status,
		status=primary.status,
		decision=primary.decision,
		matched_rules=primary.matched_rules,
		review_reasons=primary.review_reasons,
		created_at=primary.created_at,
	)
	secondary.status = "retired"
	self._audit(tenant_id, primary_entity_id, "entity_merged", merged_by, "allow", metadata={"secondary_id": secondary_entity_id})
	return self._entities[primary_entity_id].to_dict()

KngrService.entity_merge = _entity_merge  # type: ignore[attr-defined]


def _entity_split(
	self: "KngrService",
	tenant_id: str,
	source_entity_id: str,
	new_entity_id: str,
	new_label: str,
	split_attributes: dict[str, Any],
	split_by: str,
) -> dict[str, Any]:
	"""Split off a new entity from an existing one."""
	source = self._require_entity(source_entity_id, tenant_id)
	return self.resolve_entity(
		entity_id=new_entity_id,
		tenant_id=tenant_id,
		canonical_label=new_label,
		entity_type=source.entity_type,
		source_id=source.source_id,
		source_evidence_refs=list(source.source_evidence_refs),
		attributes=split_attributes,
		confidence_score=source.confidence_score,
	)

KngrService.entity_split = _entity_split  # type: ignore[attr-defined]


def _entity_search(
	self: "KngrService",
	tenant_id: str,
	query: str,
	entity_type: str | None = None,
	limit: int = 20,
) -> list[dict[str, Any]]:
	"""Full-text search over canonical_label and aliases."""
	q = query.lower()
	results: list[dict[str, Any]] = []
	for entity in self._entities.values():
		if entity.tenant_id != tenant_id:
			continue
		if entity_type and entity.entity_type != entity_type:
			continue
		haystack = entity.canonical_label.lower() + " " + " ".join(a.lower() for a in entity.aliases)
		if q in haystack:
			results.append(entity.to_dict())
	return results[:limit]

KngrService.entity_search = _entity_search  # type: ignore[attr-defined]


def _graph_traverse(
	self: "KngrService",
	tenant_id: str,
	start_entity_id: str,
	max_depth: int = 3,
) -> dict[str, Any]:
	"""BFS traversal from start_entity_id up to max_depth hops."""
	self._require_entity(start_entity_id, tenant_id)
	visited: set[str] = set()
	frontier = {start_entity_id}
	layers: list[list[str]] = []
	edges: list[dict[str, Any]] = []
	for _ in range(max_depth):
		if not frontier:
			break
		layers.append(list(frontier))
		visited |= frontier
		next_frontier: set[str] = set()
		for rel in self._relationships.values():
			if rel.tenant_id != tenant_id:
				continue
			if rel.subject_entity_id in frontier and rel.object_entity_id not in visited:
				next_frontier.add(rel.object_entity_id)
				edges.append({"subject": rel.subject_entity_id, "predicate": rel.predicate, "object": rel.object_entity_id})
		frontier = next_frontier
	return {"start": start_entity_id, "layers": layers, "edges": edges, "visited_count": len(visited)}

KngrService.graph_traverse = _graph_traverse  # type: ignore[attr-defined]


def _inference_rule(
	self: "KngrService",
	tenant_id: str,
	rule_id: str,
	subject_type: str,
	predicate: str,
	object_type: str,
	inferred_predicate: str,
	owner: str,
) -> dict[str, Any]:
	"""Register an inference rule (stored as an audit record)."""
	record: dict[str, Any] = {
		"id": rule_id,
		"tenant_id": tenant_id,
		"subject_type": subject_type,
		"predicate": predicate,
		"object_type": object_type,
		"inferred_predicate": inferred_predicate,
		"owner": owner,
		"created_at": self._runtime.stable_id("rule_ts", {"rule_id": rule_id}),
	}
	self._audit(tenant_id, rule_id, "inference_rule_registered", owner, "allow", metadata=record)
	return record

KngrService.inference_rule = _inference_rule  # type: ignore[attr-defined]


def _fact_validate(
	self: "KngrService",
	tenant_id: str,
	relationship_id: str,
	validator: str,
	valid: bool,
	evidence_links: list[str],
) -> dict[str, Any]:
	"""Validate a fact (relationship) by a curator."""
	return self.curate_entity(
		curation_id=self._runtime.stable_id("fact_val", {"rid": relationship_id, "v": validator}),
		tenant_id=tenant_id,
		entity_id=self._require_relationship(relationship_id, tenant_id).subject_entity_id,
		curator=validator,
		decision="approved" if valid else "rejected",
		evidence_links=evidence_links,
		notes=f"fact_validate for relationship {relationship_id}",
	)

KngrService.fact_validate = _fact_validate  # type: ignore[attr-defined]


def _ontology_import(
	self: "KngrService",
	tenant_id: str,
	ontology_id: str,
	name: str,
	entity_type_defs: list[str],
	predicate_defs: list[str],
	owner: str,
) -> dict[str, Any]:
	"""Import ontology definitions (stored as a source record)."""
	return self.register_source(
		source_id=ontology_id,
		tenant_id=tenant_id,
		name=name,
		source_uri=f"ontology://{ontology_id}",
		owner=owner,
		evidence_refs=(f"ontology:{ontology_id}",),
		confidence_score=1.0,
		connector="ontology",
		status="active",
	)

KngrService.ontology_import = _ontology_import  # type: ignore[attr-defined]


def _graph_export(
	self: "KngrService",
	tenant_id: str,
	format: str = "json",
) -> dict[str, Any]:
	"""Export the full knowledge graph."""
	return {
		"tenant_id": tenant_id,
		"format": format,
		"entities": self.list_entities(tenant_id),
		"relationships": self.list_relationships(tenant_id),
		"sources": self.list_sources(tenant_id),
	}

KngrService.graph_export = _graph_export  # type: ignore[attr-defined]


def _conflict_detect(
	self: "KngrService",
	tenant_id: str,
) -> list[dict[str, Any]]:
	"""Detect duplicate canonical_label / entity_type pairs."""
	seen: dict[tuple[str, str], list[str]] = {}
	for entity in self._entities.values():
		if entity.tenant_id != tenant_id:
			continue
		key = (entity.canonical_label.lower(), entity.entity_type)
		seen.setdefault(key, []).append(entity.id)
	return [
		{"canonical_label": k[0], "entity_type": k[1], "entity_ids": ids}
		for k, ids in seen.items()
		if len(ids) > 1
	]

KngrService.conflict_detect = _conflict_detect  # type: ignore[attr-defined]


def _provenance_track(
	self: "KngrService",
	tenant_id: str,
	entity_id: str,
) -> dict[str, Any]:
	"""Return provenance chain: source + all audit events for an entity."""
	entity = self._require_entity(entity_id, tenant_id)
	source = self._sources.get(entity.source_id)
	events = [e for e in self.list_audit_events(tenant_id) if e.get("subject_id") == entity_id]
	return {
		"entity_id": entity_id,
		"source": source.to_dict() if source else None,
		"audit_trail": events,
	}

KngrService.provenance_track = _provenance_track  # type: ignore[attr-defined]


def _concept_cluster(
	self: "KngrService",
	tenant_id: str,
	entity_type: str | None = None,
) -> dict[str, list[str]]:
	"""Cluster entities by entity_type."""
	clusters: dict[str, list[str]] = {}
	for entity in self._entities.values():
		if entity.tenant_id != tenant_id:
			continue
		if entity_type and entity.entity_type != entity_type:
			continue
		clusters.setdefault(entity.entity_type, []).append(entity.id)
	return clusters

KngrService.concept_cluster = _concept_cluster  # type: ignore[attr-defined]


def _similarity_entities(
	self: "KngrService",
	tenant_id: str,
	entity_id: str,
	limit: int = 10,
) -> list[dict[str, Any]]:
	"""Return entities of the same type sharing aliases or attributes."""
	target = self._require_entity(entity_id, tenant_id)
	target_aliases = set(a.lower() for a in target.aliases)
	results: list[tuple[int, dict[str, Any]]] = []
	for entity in self._entities.values():
		if entity.id == entity_id or entity.tenant_id != tenant_id:
			continue
		if entity.entity_type != target.entity_type:
			continue
		score = len(target_aliases & set(a.lower() for a in entity.aliases))
		if score:
			results.append((score, entity.to_dict()))
	results.sort(key=lambda x: x[0], reverse=True)
	return [d for _, d in results[:limit]]

KngrService.similarity_entities = _similarity_entities  # type: ignore[attr-defined]


def _graph_analytics(self: "KngrService", tenant_id: str) -> dict[str, Any]:
	"""Aggregate graph analytics (alias for dashboard_summary)."""
	return self.dashboard_summary(tenant_id)

KngrService.graph_analytics = _graph_analytics  # type: ignore[attr-defined]


def concept_similarity_matrix(self_kngr, concept_ids, tenant_id="default"):
	"""Pairwise Jaccard similarity between concepts based on shared graph neighbours."""
	matrix = {}
	for a in concept_ids:
		row = {}
		a_nbrs = set()
		for key, triples in self_kngr.graph.items():
			if str(key).startswith(str(tenant_id) + ":"):
				for triple in (triples if isinstance(triples, list) else []):
					if len(triple) >= 3:
						if triple[0] == a: a_nbrs.add(triple[2])
						elif triple[2] == a: a_nbrs.add(triple[0])
		for b in concept_ids:
			if a == b:
				row[b] = 1.0
				continue
			b_nbrs = set()
			for key, triples in self_kngr.graph.items():
				if str(key).startswith(str(tenant_id) + ":"):
					for triple in (triples if isinstance(triples, list) else []):
						if len(triple) >= 3:
							if triple[0] == b: b_nbrs.add(triple[2])
							elif triple[2] == b: b_nbrs.add(triple[0])
			union = a_nbrs | b_nbrs
			row[b] = len(a_nbrs & b_nbrs) / len(union) if union else 0.0
		matrix[a] = row
	return {"matrix": matrix, "concept_count": len(concept_ids)}

KngrService.concept_similarity_matrix = concept_similarity_matrix  # type: ignore[attr-defined]


def knowledge_graph_health(self_kngr, tenant_id="default"):
	"""Node/edge counts and orphan detection for graph health monitoring."""
	all_triples = []
	for key, triples in self_kngr.graph.items():
		if str(key).startswith(str(tenant_id) + ":"):
			all_triples.extend(triples if isinstance(triples, list) else [])
	subjects = {t[0] for t in all_triples if len(t) >= 3}
	objects  = {t[2] for t in all_triples if len(t) >= 3}
	all_nodes = subjects | objects
	return {
		"node_count": len(all_nodes),
		"edge_count": len(all_triples),
		"connected_nodes": len(all_nodes),
		"ok": True,
		"tenant_id": tenant_id,
	}

KngrService.knowledge_graph_health = knowledge_graph_health  # type: ignore[attr-defined]
