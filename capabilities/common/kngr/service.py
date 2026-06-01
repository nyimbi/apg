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
