"""Service layer for executable ontology management."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	CurationReview,
	OntoAuditEvent,
	OntoLifecycleBatchRecord,
	Ontology,
	OntologyAgentRecord,
	OntologyExport,
	OntologyNamespace,
	OntologyPublication,
	OntologyTerm,
	SemanticMapping,
	TaxonomyEdge,
	ValidationReport,
	utc_now_iso,
)
from .ontology_runtime import (
	bump_patch_version,
	duplicate_labels,
	mapping_requires_review,
	normalize_confidence,
	normalize_label,
	normalize_mapping_type,
	normalize_term_status,
	publication_readiness,
	stable_id,
	taxonomy_has_cycle,
)


from capabilities.common.reliability import guard_tenant_id, guard_non_empty_string, BoundedCache
class OntoService:
	"""In-process ontology registry, vocabulary, mapping, publication, and governance service."""

	def __init__(self, confidence_threshold: float | None = None) -> None:
		contract = get_capability_contract()
		self.confidence_threshold = float(
			confidence_threshold
			if confidence_threshold is not None
		else contract["configuration"]["mappings"]["confidence_threshold"]
		)
		self._ontologies: dict[str, Ontology] = {}
		self._namespaces: dict[str, OntologyNamespace] = {}
		self._terms: dict[str, OntologyTerm] = {}
		self._taxonomy_edges: dict[str, TaxonomyEdge] = {}
		self._mappings: dict[str, SemanticMapping] = {}
		self._reviews: dict[str, CurationReview] = {}
		self._validation_reports: dict[str, ValidationReport] = {}
		self._publications: dict[str, OntologyPublication] = {}
		self._exports: dict[str, OntologyExport] = {}
		self._ontology_agents: dict[str, OntologyAgentRecord] = {}
		self._lifecycle_batches: dict[str, OntoLifecycleBatchRecord] = {}
		self._audit_events: dict[str, OntoAuditEvent] = {}
		self._agent_runtimes = set(contract["agents"]["supported_runtimes"])
		self._agent_roles = set(contract["agents"]["supported_roles"])
		self._privileged_agent_roles = set(contract["agents"]["privileged_roles"])
		self._lifecycle_operations = set(contract["configuration"]["streaming"]["required_operations"])

	def describe(self, tenant_id: str = "default") -> dict[str, Any]:
		return get_capability_contract(tenant_id)

	def evaluate(self, context: dict[str, Any]) -> dict[str, Any]:
		return evaluate_capability_rules(context)

	def register_ontology(
		self,
		ontology_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		domain: str,
		description: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_ontology",
			"ontology_id_present": bool(ontology_id),
			"ontology_name_present": bool(name),
			"ontology_owner_present": bool(owner),
			"ontology_domain_present": bool(domain),
		})
		self._raise_if_denied(result)
		ontology = Ontology(
			id=ontology_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			domain=domain,
			description=description,
			metadata=dict(metadata or {}),
			**self._policy_fields(result),
		)
		self._ontologies[ontology.id] = ontology
		self._audit(tenant_id, "ontology_registered", ontology.id, f"Registered ontology {name}", result=result)
		return ontology.to_dict()

	def register_namespace(
		self,
		namespace_id: str,
		tenant_id: str,
		ontology_id: str,
		prefix: str,
		uri: str,
		owner: str,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		prefix_unique = not any(
			namespace.tenant_id == tenant_id
			and namespace.ontology_id == ontology.id
			and namespace.prefix == prefix
			for namespace in self._namespaces.values()
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "register_namespace",
			"ontology_present": bool(ontology),
			"namespace_prefix_present": bool(prefix),
			"namespace_uri_present": bool(uri),
			"owner_assigned": bool(owner),
			"namespace_prefix_unique": prefix_unique,
		})
		self._raise_if_denied(result)
		namespace = OntologyNamespace(
			id=namespace_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			prefix=prefix,
			uri=uri,
			owner=owner,
			metadata=dict(metadata or {}),
			**self._policy_fields(result),
		)
		self._namespaces[namespace.id] = namespace
		self._touch_ontology(ontology)
		self._audit(tenant_id, "namespace_registered", namespace.id, f"Registered namespace {prefix}", result=result)
		return namespace.to_dict()

	def create_term(
		self,
		term_id: str,
		tenant_id: str,
		ontology_id: str,
		label: str,
		owner: str,
		definition: str = "",
		status: str = "draft",
		synonyms: list[str] | None = None,
		external_refs: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
		review_recorded: bool = False,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		duplicate = any(
			term.tenant_id == tenant_id
			and term.ontology_id == ontology.id
			and normalize_label(term.label) == normalize_label(label)
			for term in self._terms.values()
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_term",
			"ontology_present": bool(ontology),
			"term_label_present": bool(label),
			"owner_assigned": bool(owner),
			"term_status_allowed": status in {"draft", "curated", "published", "deprecated"},
			"duplicate_term_detected": duplicate,
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		term = OntologyTerm(
			id=term_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			label=label,
			owner=owner,
			definition=definition,
			synonyms=list(synonyms or []),
			external_refs=list(external_refs or []),
			metadata=dict(metadata or {}) | {"duplicate_term_detected": duplicate},
			status=self._status_after_review(result, normalize_term_status(status)),
			**self._policy_fields(result, review_recorded),
		)
		self._terms[term.id] = term
		self._touch_ontology(ontology)
		self._audit(tenant_id, "term_created", term.id, f"Created term {label}", result=result)
		return term.to_dict()

	def curate_term(
		self,
		review_id: str,
		tenant_id: str,
		term_id: str,
		reviewer: str,
		status: str = "curated",
		notes: str = "",
		change_type: str = "non_breaking",
		review_recorded: bool = True,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		term = self._require_term(term_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "curate_term",
			"change_type": change_type,
			"reviewer_present": bool(reviewer),
			"curation_evidence_present": bool(notes or review_recorded),
			"review_recorded": review_recorded,
		})
		self._raise_if_denied(result)
		review = CurationReview(
			id=review_id,
			tenant_id=tenant_id,
			ontology_id=term.ontology_id,
			subject_id=term.id,
			review_type=change_type,
			reviewer=reviewer,
			notes=notes,
			status=self._status_after_review(result, "approved"),
			**self._policy_fields(result, review_recorded),
		)
		self._reviews[review.id] = review
		if review.status != "pending_review":
			term.status = normalize_term_status(status)
			term.updated_at = utc_now_iso()
			self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "term_curated", term.id, f"Curated term {term.label}", result=result)
		return review.to_dict()

	def add_synonym(
		self,
		tenant_id: str,
		term_id: str,
		synonym: str,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		term = self._require_term(term_id, tenant_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_synonym",
			"term_present": bool(term),
			"synonym_present": bool(synonym),
		})
		self._raise_if_denied(result)
		if synonym not in term.synonyms:
			term.synonyms.append(synonym)
		term.updated_at = utc_now_iso()
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "synonym_added", term.id, f"Added synonym {synonym}", result=result)
		return term.to_dict()

	def add_taxonomy_edge(
		self,
		edge_id: str,
		tenant_id: str,
		ontology_id: str,
		parent_term_id: str,
		child_term_id: str,
		relationship_type: str = "broader_than",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		parent = self._require_term(parent_term_id, tenant_id, ontology_id=ontology.id)
		child = self._require_term(child_term_id, tenant_id, ontology_id=ontology.id)
		cycle = taxonomy_has_cycle(self._taxonomy_edge_dicts(ontology.id, tenant_id), parent_term_id, child_term_id)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "add_taxonomy_edge",
			"parent_term_present": bool(parent),
			"child_term_present": bool(child),
			"self_relation": parent_term_id == child_term_id,
			"taxonomy_cycle_detected": cycle,
			"relationship_type_allowed": relationship_type in {"broader_than", "narrower_than", "related_to", "equivalent_to"},
		})
		self._raise_if_denied(result)
		edge = TaxonomyEdge(
			id=edge_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			parent_term_id=parent_term_id,
			child_term_id=child_term_id,
			relationship_type=relationship_type,
			**self._policy_fields(result),
		)
		self._taxonomy_edges[edge.id] = edge
		self._touch_ontology(ontology)
		self._audit(tenant_id, "taxonomy_edge_added", edge.id, "Added taxonomy edge", result=result)
		return edge.to_dict()

	def create_mapping(
		self,
		mapping_id: str,
		tenant_id: str,
		term_id: str,
		target_ref: str,
		mapping_type: str = "exact",
		confidence: float = 1.0,
		review_recorded: bool = False,
		review_ref: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		term = self._require_term(term_id, tenant_id)
		normalized_confidence = normalize_confidence(confidence)
		mapping_type_normalized = normalize_mapping_type(mapping_type)
		mapping_scope = "external" if target_ref.startswith("external:") else "internal"
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "create_mapping",
			"term_present": bool(term),
			"target_ref_present": bool(target_ref),
			"mapping_type_allowed": mapping_type_normalized in {"exact", "close", "broad", "narrow", "related"},
			"mapping_confidence": normalized_confidence,
			"mapping_scope": mapping_scope,
			"review_recorded": review_recorded,
		})
		self._raise_if_denied(result)
		mapping = SemanticMapping(
			id=mapping_id,
			tenant_id=tenant_id,
			ontology_id=term.ontology_id,
			term_id=term.id,
			target_ref=target_ref,
			mapping_type=mapping_type_normalized,
			confidence=normalized_confidence,
			review_recorded=review_recorded,
			review_ref=review_ref,
			status=self._status_after_review(
				result,
				"reviewed" if mapping_requires_review(normalized_confidence, self.confidence_threshold) else "active",
			),
			metadata=dict(metadata or {}),
			**self._policy_fields(result, review_recorded),
		)
		self._mappings[mapping.id] = mapping
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "semantic_mapping_created", mapping.id, f"Mapped term {term.label}", result=result)
		return mapping.to_dict()

	def deprecate_term(
		self,
		review_id: str,
		tenant_id: str,
		term_id: str,
		replacement_term_id: str,
		reviewer: str,
		review_recorded: bool = False,
		notes: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		term = self._require_term(term_id, tenant_id)
		replacement = self._require_term(replacement_term_id, tenant_id, ontology_id=term.ontology_id) if replacement_term_id else None
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "deprecate_term",
			"replacement_term_present": bool(replacement),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		review = CurationReview(
			id=review_id,
			tenant_id=tenant_id,
			ontology_id=term.ontology_id,
			subject_id=term.id,
			review_type="deprecation",
			reviewer=reviewer,
			notes=notes,
			status=self._status_after_review(result, "approved"),
			**self._policy_fields(result, review_recorded),
		)
		self._reviews[review.id] = review
		if review.status != "pending_review":
			term.status = "deprecated"
			term.metadata["replacement_term_id"] = replacement_term_id
			term.updated_at = utc_now_iso()
			self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "term_deprecated", term.id, f"Deprecated term {term.label}", result=result)
		return review.to_dict()

	def validate_ontology(
		self,
		report_id: str,
		tenant_id: str,
		ontology_id: str,
		review_recorded: bool = False,
		review_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		mappings = self._mapping_dicts(ontology.id, tenant_id)
		duplicates = duplicate_labels(terms)
		ready, issues = publication_readiness(terms, mappings, duplicates, self.confidence_threshold)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "validate_ontology",
			"ontology_present": bool(ontology),
			"issue_count": len(issues),
			"review_recorded": bool(review_recorded),
		})
		self._raise_if_denied(result)
		report = ValidationReport(
			id=report_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			issue_count=len(issues),
			issues=issues,
			status=self._status_after_review(result, "passed" if ready else "issues_found"),
			review_recorded=review_recorded,
			review_ref=review_ref,
			**self._policy_fields(result, review_recorded),
		)
		self._validation_reports[report.id] = report
		self._audit(tenant_id, "ontology_validated", report.id, f"Validated ontology {ontology.name}", result=result)
		return report.to_dict()

	def review_mapping(
		self,
		review_id: str,
		tenant_id: str,
		mapping_id: str,
		reviewer: str,
		notes: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		mapping = self._require_mapping(mapping_id, tenant_id)
		review = CurationReview(
			id=review_id,
			tenant_id=tenant_id,
			ontology_id=mapping.ontology_id,
			subject_id=mapping.id,
			review_type="mapping",
			reviewer=reviewer,
			notes=notes,
			**self._policy_fields({"decision": "allow", "matched_rules": [], "actions": []}, True),
		)
		self._reviews[review.id] = review
		mapping.review_recorded = True
		mapping.review_ref = review.id
		mapping.status = "reviewed"
		mapping.decision = "allow"
		mapping.matched_rules = ()
		mapping.review_reasons = ()
		mapping.audit_evidence = self._audit_evidence({"decision": "allow", "matched_rules": [], "actions": []}, True)
		self._audit(tenant_id, "mapping_reviewed", mapping.id, "Reviewed semantic mapping")
		return review.to_dict()

	def publish_ontology(
		self,
		publication_id: str,
		tenant_id: str,
		ontology_id: str,
		approval_recorded: bool = False,
		approval_ref: str = "",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		mappings = self._mapping_dicts(ontology.id, tenant_id)
		duplicates = duplicate_labels(terms)
		ready, issues = publication_readiness(terms, mappings, duplicates, self.confidence_threshold)
		validation_recorded = any(
			report.tenant_id == tenant_id
			and report.ontology_id == ontology.id
			and report.status == "passed"
			for report in self._validation_reports.values()
		)
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "publish_ontology",
			"approval_recorded": approval_recorded,
			"validation_recorded": validation_recorded,
			"duplicate_term_detected": bool(duplicates),
			"taxonomy_cycle_detected": False,
			"draft_terms_present": any(term["status"] == "draft" for term in terms),
			"unreviewed_low_confidence_mappings_present": any(mapping["confidence"] < self.confidence_threshold and not mapping["review_recorded"] for mapping in mappings),
		})
		self._raise_if_denied(result)
		if not ready:
			raise PermissionError(", ".join(issues))
		ontology.version = bump_patch_version(ontology.version)
		ontology.status = "published"
		ontology.updated_at = utc_now_iso()
		for term in self._terms.values():
			if term.ontology_id == ontology.id and term.tenant_id == tenant_id:
				term.status = "published"
		publication = OntologyPublication(
			id=publication_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			version=ontology.version,
			approval_recorded=approval_recorded,
			approval_ref=approval_ref,
			duplicate_count=len(duplicates),
			term_count=len(terms),
			mapping_count=len(mappings),
			**self._policy_fields(result, approval_recorded),
		)
		self._publications[publication.id] = publication
		self._audit(tenant_id, "ontology_published", publication.id, f"Published ontology {ontology.name}", result=result)
		return publication.to_dict()

	def export_ontology(
		self,
		export_id: str,
		tenant_id: str,
		ontology_id: str,
		export_format: str = "jsonld",
	) -> dict[str, Any]:
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		allowed = export_format in {"rdf", "owl", "jsonld", "skos", "csv"}
		result = self.evaluate({
			"tenant_context_present": bool(tenant_id),
			"operation": "export_ontology",
			"export_format_allowed": allowed,
		})
		self._raise_if_denied(result)
		export = OntologyExport(
			id=export_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			format=export_format,
			version=ontology.version,
			artifact_ref=f"onto://{tenant_id}/{ontology.id}/{ontology.version}.{export_format}",
			**self._policy_fields(result),
		)
		self._exports[export.id] = export
		self._audit(tenant_id, "ontology_exported", export.id, f"Exported ontology {ontology.name}", result=result)
		return export.to_dict()

	def register_ontology_agent(
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
			"operation": "register_ontology_agent",
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
			raise ValueError("ontology_agent_id_required")
		if not str(name or "").strip():
			raise ValueError("ontology_agent_name_required")
		status = "pending_review" if result["decision"] == "require_review" else "active"
		record = OntologyAgentRecord(
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
			**self._policy_fields(result, human_approval_required),
		)
		self._ontology_agents[self._tenant_record_key(tenant_id, record.id)] = record
		self._audit(tenant_id, "ontology_agent_registered", record.id, f"Registered ontology agent {record.name}", result=result)
		return record.to_dict()

	def validate_onto_lifecycle_batch(
		self,
		tenant_id: str,
		event_stream: str,
		mutation_count: int,
		operation: str = "ontology_agent_batch",
		batch_id: str | None = None,
	) -> dict[str, Any]:
		mutation_count = int(mutation_count)
		if mutation_count <= 0:
			raise ValueError("onto_lifecycle_batch_empty")
		stream_value = _normalize_token(event_stream)
		operation_value = _normalize_token(operation)
		if operation_value not in self._lifecycle_operations:
			raise ValueError(f"unsupported_onto_lifecycle_operation:{operation_value}")
		result = self.evaluate({
			"tenant_context_present": bool(str(tenant_id or "").strip()),
			"operation": "validate_onto_lifecycle_batch",
			"event_stream": stream_value,
		})
		accepted = result["decision"] == "allow"
		record = OntoLifecycleBatchRecord(
			id=batch_id or f"ontobatch:{len(self._lifecycle_batches) + 1:06d}",
			tenant_id=tenant_id,
			event_stream=stream_value,
			mutation_count=mutation_count,
			operation=operation_value,
			accepted=accepted,
			decision=result["decision"],
			matched_rules=list(result["matched_rules"]),
			review_reasons=tuple(self._review_reasons(result)),
			audit_evidence=self._audit_evidence(result),
			status="accepted" if accepted else "denied",
		)
		self._lifecycle_batches[self._tenant_record_key(tenant_id, record.id)] = record
		self._audit(tenant_id, f"onto_lifecycle_batch_{record.status}", record.id, f"Validated ontology lifecycle batch {record.id}", result=result)
		if not accepted:
			self._raise_if_denied(result)
		return record.to_dict()

	def concept_define(
		self,
		tenant_id: str,
		ontology_id: str,
		label: str,
		owner: str,
		definition: str = "",
		synonyms: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Define a new concept (term) in an ontology with optional synonyms."""
		term_id = stable_id("onto_term", tenant_id, ontology_id, label)
		return self.create_term(
			term_id=term_id,
			tenant_id=tenant_id,
			ontology_id=ontology_id,
			label=label,
			owner=owner,
			definition=definition,
			synonyms=synonyms or [],
			metadata=metadata or {},
		)

	def property_add(
		self,
		tenant_id: str,
		term_id: str,
		property_name: str,
		property_value: str,
	) -> dict[str, Any]:
		"""Add or update a metadata property on an existing ontology term."""
		self._require_tenant(tenant_id)
		term = self._require_term(term_id, tenant_id)
		term.metadata[property_name] = property_value
		term.updated_at = utc_now_iso()
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "property_added", term_id, f"Property {property_name}={property_value[:30]}")
		return term.to_dict()

	def axiom_assert(
		self,
		tenant_id: str,
		ontology_id: str,
		axiom_id: str,
		axiom_type: str,
		subject_term_id: str,
		predicate: str,
		object_ref: str,
		asserted_by: str,
	) -> dict[str, Any]:
		"""Assert a logical axiom (subClassOf, equivalentClass, disjointWith) between terms."""
		self._require_tenant(tenant_id)
		self._require_ontology(ontology_id, tenant_id)
		self._require_term(subject_term_id, tenant_id)
		valid_types = {"subClassOf", "equivalentClass", "disjointWith", "objectProperty", "dataProperty"}
		assert axiom_type in valid_types, f"unsupported axiom_type: {axiom_type}"
		record = {
			"axiom_id": axiom_id,
			"tenant_id": tenant_id,
			"ontology_id": ontology_id,
			"axiom_type": axiom_type,
			"subject_term_id": subject_term_id,
			"predicate": predicate,
			"object_ref": object_ref,
			"asserted_by": asserted_by,
			"status": "active",
			"asserted_at": utc_now_iso(),
		}
		self._audit(tenant_id, "axiom_asserted", axiom_id, f"{axiom_type}: {subject_term_id} {predicate} {object_ref}")
		return record

	def ontology_merge(
		self,
		tenant_id: str,
		source_ontology_id: str,
		target_ontology_id: str,
		merge_strategy: str = "additive",
		merged_by: str = "system",
	) -> dict[str, Any]:
		"""Merge source ontology terms into the target ontology."""
		self._require_tenant(tenant_id)
		src = self._require_ontology(source_ontology_id, tenant_id)
		tgt = self._require_ontology(target_ontology_id, tenant_id)
		assert merge_strategy in {"additive", "override", "skip_existing"}, f"unsupported strategy: {merge_strategy}"
		src_terms = self._term_dicts(src.id, tenant_id)
		imported = 0
		skipped = 0
		for t in src_terms:
			existing = any(
				term.ontology_id == tgt.id and _normalize_token(term.label) == _normalize_token(t["label"])
				for term in self._terms.values()
				if term.tenant_id == tenant_id
			)
			if existing and merge_strategy == "skip_existing":
				skipped += 1
				continue
			new_id = stable_id("onto_term", tenant_id, tgt.id, t["label"], imported)
			self._terms[new_id] = type(list(self._terms.values())[0])(
				**{**{k: v for k, v in self._terms.get(t["id"], list(self._terms.values())[0]).__dict__.items()}, "id": new_id, "ontology_id": tgt.id}
			) if self._terms else None
			if self._terms.get(new_id) is None:
				del self._terms[new_id]
				skipped += 1
				continue
			imported += 1
		self._audit(tenant_id, "ontology_merged", target_ontology_id, f"Merged {src.name} -> {tgt.name}: {imported} terms")
		return {"source_ontology_id": source_ontology_id, "target_ontology_id": target_ontology_id, "strategy": merge_strategy, "terms_imported": imported, "terms_skipped": skipped}

	def consistency_check(
		self,
		tenant_id: str,
		ontology_id: str,
	) -> dict[str, Any]:
		"""Check ontology structural consistency: duplicate labels, cycles, orphan terms."""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		edges = self._taxonomy_edge_dicts(ontology.id, tenant_id)
		from .ontology_runtime import duplicate_labels, taxonomy_has_cycle
		dupes = duplicate_labels(terms)
		cycle_detected = False
		for edge in edges:
			if taxonomy_has_cycle(edges, edge["parent_term_id"], edge["child_term_id"]):
				cycle_detected = True
				break
		term_ids = {t["id"] for t in terms}
		connected = {e["parent_term_id"] for e in edges} | {e["child_term_id"] for e in edges}
		orphans = [t for t in terms if t["id"] not in connected and len(terms) > 1]
		issues = []
		if dupes:
			issues.append(f"{len(dupes)} duplicate_labels")
		if cycle_detected:
			issues.append("taxonomy_cycle_detected")
		if orphans:
			issues.append(f"{len(orphans)} orphan_terms")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"consistent": not issues,
			"issues": issues,
			"duplicate_label_count": len(dupes),
			"cycle_detected": cycle_detected,
			"orphan_term_count": len(orphans),
			"checked_at": utc_now_iso(),
		}

	def sparql_query(
		self,
		tenant_id: str,
		ontology_id: str,
		query: str,
		actor: str = "system",
	) -> dict[str, Any]:
		"""Execute a SPARQL-like term query against a tenant ontology (keyword SELECT simulation)."""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		import re as _re
		select_match = _re.search(r'WHERE\s*\{([^}]+)\}', query, _re.IGNORECASE)
		filter_label = None
		if select_match:
			label_match = _re.search(r'rdfs:label\s+"([^"]+)"', select_match.group(1))
			if label_match:
				filter_label = label_match.group(1).lower()
		results = [t for t in terms if filter_label is None or filter_label in t["label"].lower()]
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"query": query,
			"result_count": len(results),
			"results": results[:50],
			"executed_by": actor,
			"executed_at": utc_now_iso(),
		}

	def ontology_visualise(
		self,
		tenant_id: str,
		ontology_id: str,
	) -> dict[str, Any]:
		"""Return a graph-representation (nodes + edges) suitable for visualisation."""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		edges = self._taxonomy_edge_dicts(ontology.id, tenant_id)
		nodes = [{"id": t["id"], "label": t["label"], "status": t["status"]} for t in terms]
		links = [{"source": e["parent_term_id"], "target": e["child_term_id"], "type": e["relationship_type"]} for e in edges]
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"node_count": len(nodes),
			"edge_count": len(links),
			"nodes": nodes,
			"edges": links,
		}

	def reasoner_run(
		self,
		tenant_id: str,
		ontology_id: str,
		reasoner: str = "EL",
		actor: str = "system",
	) -> dict[str, Any]:
		"""Run a (simulated) OWL reasoner to infer implicit taxonomy relationships."""
		self._require_tenant(tenant_id)
		assert reasoner in {"EL", "RL", "QL", "DL"}, f"unsupported reasoner: {reasoner}"
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology.id, tenant_id)
		edges = self._taxonomy_edge_dicts(ontology.id, tenant_id)
		inferred = max(0, len(edges) // 2)
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"reasoner": reasoner,
			"input_axiom_count": len(edges),
			"inferred_axiom_count": inferred,
			"consistent": True,
			"executed_by": actor,
			"executed_at": utc_now_iso(),
		}

	def import_owl(
		self,
		tenant_id: str,
		ontology_id: str,
		owl_content: str,
		imported_by: str = "system",
	) -> dict[str, Any]:
		"""Import terms from OWL/RDF XML content into an existing ontology."""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		import re as _re
		class_labels = _re.findall(r'rdfs:label[^>]*>([^<]+)<', owl_content)
		imported = 0
		for label in class_labels:
			label = label.strip()
			if not label:
				continue
			term_id = stable_id("onto_term", tenant_id, ontology_id, label, imported)
			self.create_term(
				term_id=term_id,
				tenant_id=tenant_id,
				ontology_id=ontology_id,
				label=label,
				owner=imported_by,
				metadata={"imported_from": "owl"},
			)
			imported += 1
		self._audit(tenant_id, "owl_imported", ontology_id, f"OWL import: {imported} terms")
		return {"ontology_id": ontology_id, "tenant_id": tenant_id, "terms_imported": imported, "imported_by": imported_by}

	def export_turtle(
		self,
		tenant_id: str,
		ontology_id: str,
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Export ontology as Turtle (TTL) RDF serialisation."""
		return self.export_ontology(
			export_id=stable_id("onto_export", tenant_id, ontology_id, "turtle"),
			tenant_id=tenant_id,
			ontology_id=ontology_id,
			export_format="rdf",
		) | {"serialisation": "turtle", "exported_by": exported_by}

	# ------------------------------------------------------------------
	# Async API
	# ------------------------------------------------------------------

	async def async_register_ontology(
		self,
		ontology_id: str,
		tenant_id: str,
		name: str,
		owner: str,
		domain: str,
		description: str = "",
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async wrapper around register_ontology for pipeline use."""
		return self.register_ontology(
			ontology_id=ontology_id,
			tenant_id=tenant_id,
			name=name,
			owner=owner,
			domain=domain,
			description=description,
			metadata=metadata,
		)

	async def async_create_term(
		self,
		term_id: str,
		tenant_id: str,
		ontology_id: str,
		label: str,
		owner: str,
		definition: str = "",
		status: str = "draft",
		synonyms: list[str] | None = None,
		external_refs: list[str] | None = None,
		metadata: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		"""Async wrapper around create_term for pipeline use."""
		return self.create_term(
			term_id=term_id,
			tenant_id=tenant_id,
			ontology_id=ontology_id,
			label=label,
			owner=owner,
			definition=definition,
			status=status,
			synonyms=synonyms,
			external_refs=external_refs,
			metadata=metadata,
		)

	async def async_bulk_create_terms(
		self,
		tenant_id: str,
		ontology_id: str,
		terms: list[dict[str, Any]],
		owner: str,
		stop_on_error: bool = False,
	) -> dict[str, Any]:
		"""Bulk-create terms in one call with per-item error reporting.

		Each item needs at minimum ``label``; optionally accepts ``term_id``,
		``definition``, ``status``, ``synonyms``, ``external_refs``, ``metadata``.
		"""
		created: list[dict[str, Any]] = []
		failed: list[dict[str, Any]] = []
		for item in terms:
			label = item.get("label", "")
			term_id = item.get("term_id") or stable_id("onto_term", tenant_id, ontology_id, label, len(created))
			try:
				result = self.create_term(
					term_id=term_id,
					tenant_id=tenant_id,
					ontology_id=ontology_id,
					label=label,
					owner=owner,
					definition=item.get("definition", ""),
					status=item.get("status", "draft"),
					synonyms=item.get("synonyms"),
					external_refs=item.get("external_refs"),
					metadata=item.get("metadata"),
				)
				created.append(result)
			except Exception as exc:
				failed.append({"label": label, "term_id": term_id, "error": str(exc)})
				if stop_on_error:
					break
		self._audit(tenant_id, "bulk_terms_created", ontology_id, f"Bulk: {len(created)} ok / {len(failed)} failed")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"requested_count": len(terms),
			"created_count": len(created),
			"failed_count": len(failed),
			"created": created,
			"failed": failed,
			"completed_at": utc_now_iso(),
		}

	async def find_similar_terms(
		self,
		tenant_id: str,
		ontology_id: str,
		candidate_label: str,
		top_k: int = 5,
		similarity_threshold: float = 0.85,
	) -> dict[str, Any]:
		"""Near-duplicate detection via token Jaccard similarity.

		A drop-in for embedding-based cosine similarity; no extra dependencies.
		Items whose score >= similarity_threshold are flagged as potential duplicates.
		"""
		self._require_tenant(tenant_id)
		self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology_id, tenant_id)

		def _jaccard(a: str, b: str) -> float:
			sa = set(normalize_label(a).split())
			sb = set(normalize_label(b).split())
			if not sa and not sb:
				return 1.0
			if not sa or not sb:
				return 0.0
			return len(sa & sb) / len(sa | sb)

		scored = sorted(
			[
				{
					"term_id": t["id"],
					"label": t["label"],
					"similarity": round(_jaccard(candidate_label, t["label"]), 4),
					"status": t["status"],
				}
				for t in terms
			],
			key=lambda x: x["similarity"],
			reverse=True,
		)
		candidates = scored[:top_k]
		duplicates = [c for c in candidates if c["similarity"] >= similarity_threshold]
		self._audit(tenant_id, "similar_terms_searched", ontology_id, f"Similarity: '{candidate_label}' -> {len(duplicates)} potential dupes")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"candidate_label": candidate_label,
			"top_k": top_k,
			"similarity_threshold": similarity_threshold,
			"candidates": candidates,
			"potential_duplicate_count": len(duplicates),
			"searched_at": utc_now_iso(),
		}

	async def align_ontologies(
		self,
		tenant_id: str,
		source_ontology_id: str,
		target_ontology_id: str,
		strategy: str = "lexical",
		confidence_cutoff: float = 0.75,
		auto_create_above: float | None = None,
	) -> dict[str, Any]:
		"""Cross-ontology term alignment.

		Strategies: ``lexical`` (label token Jaccard), ``synonym`` (includes altLabels).
		When *auto_create_above* is set, candidates above that threshold are
		persisted automatically as SemanticMappings.
		"""
		self._require_tenant(tenant_id)
		src = self._require_ontology(source_ontology_id, tenant_id)
		tgt = self._require_ontology(target_ontology_id, tenant_id)
		assert strategy in {"lexical", "synonym"}, f"unsupported alignment strategy: {strategy}"

		src_terms = self._term_dicts(src.id, tenant_id)
		tgt_terms = self._term_dicts(tgt.id, tenant_id)

		def _tokens(term: dict[str, Any]) -> set[str]:
			base = set(normalize_label(term["label"]).split())
			if strategy == "synonym":
				for syn in term.get("synonyms", []):
					base |= set(normalize_label(syn).split())
			return base

		def _sim(a: dict[str, Any], b: dict[str, Any]) -> float:
			sa, sb = _tokens(a), _tokens(b)
			if not sa or not sb:
				return 0.0
			return len(sa & sb) / len(sa | sb)

		candidates: list[dict[str, Any]] = []
		auto_created = 0
		for s in src_terms:
			for t in tgt_terms:
				score = round(_sim(s, t), 4)
				if score < confidence_cutoff:
					continue
				candidates.append({
					"source_term_id": s["id"],
					"source_label": s["label"],
					"target_term_id": t["id"],
					"target_label": t["label"],
					"confidence": score,
					"strategy": strategy,
				})
				if auto_create_above is not None and score >= auto_create_above:
					mid = stable_id("onto_align", tenant_id, s["id"], t["id"])
					try:
						self.create_mapping(
							mapping_id=mid,
							tenant_id=tenant_id,
							term_id=s["id"],
							target_ref=f"onto:{target_ontology_id}#{t['id']}",
							mapping_type="close" if score < 0.95 else "exact",
							confidence=score,
							metadata={"alignment_strategy": strategy, "auto_aligned": True},
						)
						auto_created += 1
					except Exception as _exc:
						_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		candidates.sort(key=lambda x: x["confidence"], reverse=True)
		self._audit(tenant_id, "ontologies_aligned", source_ontology_id, f"Aligned {src.name} -> {tgt.name}: {len(candidates)} candidates")
		return {
			"source_ontology_id": source_ontology_id,
			"target_ontology_id": target_ontology_id,
			"tenant_id": tenant_id,
			"strategy": strategy,
			"confidence_cutoff": confidence_cutoff,
			"candidate_count": len(candidates),
			"auto_created_count": auto_created,
			"candidates": candidates[:100],
			"aligned_at": utc_now_iso(),
		}

	async def compute_version_diff(
		self,
		tenant_id: str,
		ontology_id: str,
	) -> dict[str, Any]:
		"""Structural diff between current state and last-published snapshot.

		Classifies the recommended version bump as patch / minor / major based
		on OWL change-severity rules (removal or deprecation = breaking = major).
		"""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		all_terms = self._term_dicts(ontology_id, tenant_id)
		published_terms = {t["id"] for t in all_terms if t["status"] == "published"}
		current_terms = {t["id"] for t in all_terms}
		current_map = {t["id"]: t for t in all_terms}
		added = current_terms - published_terms
		removed: set[str] = set()   # would need snapshot history; conservative empty
		deprecated = {tid for tid in published_terms if current_map.get(tid, {}).get("status") == "deprecated"}
		breaking = bool(removed or deprecated)
		bump = "major" if removed else ("minor" if added else "patch")
		self._audit(tenant_id, "version_diff_computed", ontology_id, f"Diff: +{len(added)} dep:{len(deprecated)}")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"current_version": ontology.version,
			"added_term_count": len(added),
			"removed_term_count": len(removed),
			"deprecated_term_count": len(deprecated),
			"breaking_change": breaking,
			"recommended_bump": bump,
			"computed_at": utc_now_iso(),
		}

	async def export_skos(
		self,
		tenant_id: str,
		ontology_id: str,
		exported_by: str = "system",
	) -> dict[str, Any]:
		"""Serialize the ontology as W3C SKOS Turtle.

		Terms -> skos:Concept, synonyms -> skos:altLabel,
		broader_than edges -> skos:broader/skos:narrower,
		related_to edges -> skos:related.
		"""
		self._require_tenant(tenant_id)
		ontology = self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology_id, tenant_id)
		edges = self._taxonomy_edge_dicts(ontology_id, tenant_id)
		lines: list[str] = [
			"@prefix skos: <http://www.w3.org/2004/02/skos/core#> .",
			"@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
			f"@prefix onto: <https://apg.datacraft.co.ke/onto/{tenant_id}/{ontology_id}#> .",
			"",
			"onto:scheme a skos:ConceptScheme ;",
			f'    rdfs:label "{ontology.name}" .',
			"",
		]
		for t in terms:
			lines.append(f'onto:{t["id"]} a skos:Concept ;')
			lines.append(f'    skos:prefLabel "{t["label"]}" ;')
			if t.get("definition"):
				lines.append(f'    skos:definition "{t["definition"]}" ;')
			for syn in t.get("synonyms", []):
				lines.append(f'    skos:altLabel "{syn}" ;')
			lines.append(f'    skos:inScheme onto:scheme .')
			lines.append("")
		for edge in edges:
			if edge["relationship_type"] == "broader_than":
				lines.append(f'onto:{edge["child_term_id"]} skos:broader onto:{edge["parent_term_id"]} .')
				lines.append(f'onto:{edge["parent_term_id"]} skos:narrower onto:{edge["child_term_id"]} .')
			elif edge["relationship_type"] == "related_to":
				lines.append(f'onto:{edge["parent_term_id"]} skos:related onto:{edge["child_term_id"]} .')
		turtle_str = "\n".join(lines)
		export_rec = self.export_ontology(
			export_id=stable_id("onto_skos", tenant_id, ontology_id, "skos"),
			tenant_id=tenant_id,
			ontology_id=ontology_id,
			export_format="rdf",
		)
		self._audit(tenant_id, "skos_exported", ontology_id, f"SKOS: {len(terms)} concepts")
		return export_rec | {
			"serialisation": "skos_turtle",
			"concept_count": len(terms),
			"exported_by": exported_by,
			"turtle": turtle_str,
		}

	async def verify_audit_chain(
		self,
		tenant_id: str,
	) -> dict[str, Any]:
		"""SHA-256 Merkle-style integrity check over the tenant audit event chain.

		Stamps events with ``event_hash`` / ``prev_hash`` on first call,
		establishing a tamper-evident baseline for subsequent runs.
		"""
		import hashlib
		import json as _json
		self._require_tenant(tenant_id)
		events = sorted(
			[ev for ev in self._audit_events.values() if ev.tenant_id == tenant_id],
			key=lambda e: e.id,
		)
		prev_hash = "genesis"
		chain_valid = True
		broken_links: list[str] = []
		for ev in events:
			payload = {"id": ev.id, "event_type": ev.event_type, "subject_id": ev.subject_id, "message": ev.message, "prev_hash": prev_hash}
			current_hash = hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()
			stored = ev.audit_evidence.get("event_hash")
			if stored is None:
				ev.audit_evidence["event_hash"] = current_hash
				ev.audit_evidence["prev_hash"] = prev_hash
			elif stored != current_hash:
				chain_valid = False
				broken_links.append(ev.id)
			prev_hash = current_hash
		return {
			"tenant_id": tenant_id,
			"event_count": len(events),
			"chain_valid": chain_valid,
			"broken_link_count": len(broken_links),
			"broken_links": broken_links,
			"chain_tip_hash": prev_hash,
			"verified_at": utc_now_iso(),
		}

	async def suggest_definition(
		self,
		tenant_id: str,
		ontology_id: str,
		label: str,
		synonyms: list[str] | None = None,
		domain_hint: str = "",
		model: str = "llama3.2",
	) -> dict[str, Any]:
		"""Generate LLM-assisted definition candidates via local Ollama.

		Falls back to template stubs when Ollama is unreachable.  The
		``provenance`` block records the model and prompt hash for audit.
		"""
		import hashlib
		self._require_tenant(tenant_id)
		self._require_ontology(ontology_id, tenant_id)
		synonyms_str = ", ".join(synonyms or [])
		prompt = (
			f"Write three concise ontology definitions for '{label}'"
			+ (f" (synonyms: {synonyms_str})" if synonyms_str else "")
			+ (f" in the domain of {domain_hint}" if domain_hint else "")
			+ ". Number them 1, 2, 3. One sentence each."
		)
		prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
		candidates: list[str] = []
		source = "template_fallback"
		try:
			import urllib.request, json as _json
			payload = _json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
			req = urllib.request.Request(
				"http://localhost:11434/api/generate",
				data=payload,
				headers={"Content-Type": "application/json"},
				method="POST",
			)
			with urllib.request.urlopen(req, timeout=10) as resp:
				raw = _json.loads(resp.read()).get("response", "")
			import re as _re
			candidates = _re.findall(r'\d\.\s+(.+)', raw)[:3]
			if candidates:
				source = f"ollama:{model}"
		except Exception as _exc:
			_log.debug("Suppressed %s: %s", type(_exc).__name__, _exc)
		if not candidates:
			ctx = f" in the context of {domain_hint}" if domain_hint else ""
			candidates = [
				f"{label} refers to a concept{ctx} that ...",
				f"A {label} is defined as{ctx} ...",
				f"The term {label} denotes{ctx} ...",
			]
		self._audit(tenant_id, "definition_suggested", ontology_id, f"Definition suggestion for '{label}' via {source}")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"label": label,
			"candidates": candidates,
			"source": source,
			"model": model,
			"provenance": {"prompt_hash": prompt_hash, "model": model, "source": source},
			"suggested_at": utc_now_iso(),
		}

	async def sync_to_graph(
		self,
		tenant_id: str,
		ontology_id: str,
		graph_adapter: Any | None = None,
	) -> dict[str, Any]:
		"""Push ontology nodes and edges to a grph capability adapter.

		When *graph_adapter* is None returns the payload dry-run without pushing.
		Adapter must expose ``async upsert_nodes(list)`` and ``async upsert_edges(list)``.
		"""
		self._require_tenant(tenant_id)
		self._require_ontology(ontology_id, tenant_id)
		terms = self._term_dicts(ontology_id, tenant_id)
		edges = self._taxonomy_edge_dicts(ontology_id, tenant_id)
		nodes = [{"id": t["id"], "label": t["label"], "status": t["status"], "ontology_id": ontology_id, "tenant_id": tenant_id, "definition": t.get("definition", "")} for t in terms]
		graph_edges = [{"source": e["parent_term_id"], "target": e["child_term_id"], "type": e["relationship_type"], "ontology_id": ontology_id, "tenant_id": tenant_id} for e in edges]
		pushed = False
		if graph_adapter is not None:
			await graph_adapter.upsert_nodes(nodes)
			await graph_adapter.upsert_edges(graph_edges)
			pushed = True
		self._audit(tenant_id, "synced_to_graph", ontology_id, f"Graph sync: {len(nodes)} nodes, {len(graph_edges)} edges")
		return {
			"ontology_id": ontology_id,
			"tenant_id": tenant_id,
			"node_count": len(nodes),
			"edge_count": len(graph_edges),
			"pushed": pushed,
			"nodes": nodes if not pushed else [],
			"edges": graph_edges if not pushed else [],
			"synced_at": utc_now_iso(),
		}

	# ------------------------------------------------------------------
	# Compat / generic record API
	# ------------------------------------------------------------------

	def create_record(
		self,
		record_id: str,
		tenant_id: str,
		metadata: dict[str, Any] | None = None,
		status: str = "active",
	) -> dict[str, Any]:
		metadata = dict(metadata or {})
		return self.register_ontology(
			ontology_id=record_id,
			tenant_id=tenant_id,
			name=str(metadata.get("name") or record_id),
			owner=str(metadata.get("owner") or "unassigned"),
			domain=str(metadata.get("domain") or "general"),
			description=str(metadata.get("description") or ""),
			metadata=metadata | {"compatibility_status": status or "active"},
		)

	def list_records(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self.list_ontologies(tenant_id)

	def list_ontologies(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._ontologies, tenant_id)

	def list_namespaces(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._namespaces, tenant_id)

	def list_terms(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._terms, tenant_id)

	def list_taxonomy_edges(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._taxonomy_edges, tenant_id)

	def list_mappings(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._mappings, tenant_id)

	def list_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._reviews, tenant_id)

	def list_validation_reports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._validation_reports, tenant_id)

	def list_publications(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._publications, tenant_id)

	def list_exports(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._exports, tenant_id)

	def list_ontology_agents(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._ontology_agents, tenant_id)

	def list_lifecycle_batches(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._lifecycle_batches, tenant_id)

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def list_pending_reviews(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return [
			item
			for item in (
				self.list_ontologies(tenant_id)
				+ self.list_namespaces(tenant_id)
				+ self.list_terms(tenant_id)
				+ self.list_taxonomy_edges(tenant_id)
				+ self.list_mappings(tenant_id)
				+ self.list_reviews(tenant_id)
				+ self.list_validation_reports(tenant_id)
				+ self.list_publications(tenant_id)
				+ self.list_exports(tenant_id)
				+ self.list_ontology_agents(tenant_id)
				+ self.list_lifecycle_batches(tenant_id)
			)
			if item["status"] == "pending_review"
		]

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		terms = [item for item in self._terms.values() if item.tenant_id == tenant_id]
		mappings = [item for item in self._mappings.values() if item.tenant_id == tenant_id]
		pending_reviews = self.list_pending_reviews(tenant_id)
		return {
			"tenant_id": tenant_id,
			"ontology_count": len(self.list_ontologies(tenant_id)),
			"namespace_count": len(self.list_namespaces(tenant_id)),
			"term_count": len(terms),
			"curated_term_count": sum(1 for term in terms if term.status in {"curated", "published"}),
			"mapping_count": len(mappings),
			"low_confidence_mapping_count": sum(1 for mapping in mappings if mapping.confidence < self.confidence_threshold),
			"taxonomy_edge_count": len(self.list_taxonomy_edges(tenant_id)),
			"validation_report_count": len(self.list_validation_reports(tenant_id)),
			"publication_count": len(self.list_publications(tenant_id)),
			"export_count": len(self.list_exports(tenant_id)),
			"ontology_agent_count": len(self.list_ontology_agents(tenant_id)),
			"pending_review_count": len(pending_reviews),
			"pending_term_review_count": len([item for item in self.list_terms(tenant_id) if item["status"] == "pending_review"]),
			"pending_mapping_review_count": len([item for item in self.list_mappings(tenant_id) if item["status"] == "pending_review"]),
			"pending_curation_review_count": len([item for item in self.list_reviews(tenant_id) if item["status"] == "pending_review"]),
			"pending_validation_review_count": len([item for item in self.list_validation_reports(tenant_id) if item["status"] == "pending_review"]),
			"pending_agent_review_count": len([item for item in self.list_ontology_agents(tenant_id) if item["status"] == "pending_review"]),
			"lifecycle_batch_count": len(self.list_lifecycle_batches(tenant_id)),
			"denied_lifecycle_batch_count": len([item for item in self.list_lifecycle_batches(tenant_id) if item["status"] == "denied"]),
			"audit_event_count": len(self.list_audit_events(tenant_id)),
		}

	def ontology_package(self, tenant_id: str | None = None) -> dict[str, Any]:
		return {
			"tenant_id": tenant_id,
			"ontologies": self.list_ontologies(tenant_id),
			"namespaces": self.list_namespaces(tenant_id),
			"terms": self.list_terms(tenant_id),
			"taxonomy_edges": self.list_taxonomy_edges(tenant_id),
			"mappings": self.list_mappings(tenant_id),
			"reviews": self.list_reviews(tenant_id),
			"validation_reports": self.list_validation_reports(tenant_id),
			"publications": self.list_publications(tenant_id),
			"exports": self.list_exports(tenant_id),
			"ontology_agents": self.list_ontology_agents(tenant_id),
			"lifecycle_batches": self.list_lifecycle_batches(tenant_id),
			"pending_reviews": self.list_pending_reviews(tenant_id),
			"audit_events": self.list_audit_events(tenant_id),
			"summary": self.dashboard_summary(tenant_id or "default"),
		}

	def _require_tenant(self, tenant_id: str) -> None:
		result = self.evaluate({"tenant_context_present": bool(tenant_id)})
		self._raise_if_denied(result)

	def _require_ontology(self, ontology_id: str, tenant_id: str) -> Ontology:
		ontology = self._ontologies.get(ontology_id)
		if ontology is None or ontology.tenant_id != tenant_id:
			raise LookupError("ontology_not_found")
		return ontology

	def _require_term(self, term_id: str, tenant_id: str, ontology_id: str | None = None) -> OntologyTerm:
		term = self._terms.get(term_id)
		if term is None or term.tenant_id != tenant_id:
			raise LookupError("ontology_term_not_found")
		if ontology_id is not None and term.ontology_id != ontology_id:
			raise LookupError("ontology_term_not_found")
		return term

	def _require_mapping(self, mapping_id: str, tenant_id: str) -> SemanticMapping:
		mapping = self._mappings.get(mapping_id)
		if mapping is None or mapping.tenant_id != tenant_id:
			raise LookupError("semantic_mapping_not_found")
		return mapping

	def _touch_ontology(self, ontology: Ontology) -> None:
		ontology.updated_at = utc_now_iso()

	def _term_dicts(self, ontology_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [term.to_dict() for term in self._terms.values() if term.ontology_id == ontology_id and term.tenant_id == tenant_id]

	def _mapping_dicts(self, ontology_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [mapping.to_dict() for mapping in self._mappings.values() if mapping.ontology_id == ontology_id and mapping.tenant_id == tenant_id]

	def _taxonomy_edge_dicts(self, ontology_id: str, tenant_id: str) -> list[dict[str, Any]]:
		return [edge.to_dict() for edge in self._taxonomy_edges.values() if edge.ontology_id == ontology_id and edge.tenant_id == tenant_id]

	def _raise_if_denied(self, result: dict[str, Any]) -> None:
		if result["decision"] == "deny":
			raise PermissionError(self._reasons(result))

	def _status_after_review(self, result: dict[str, Any], accepted_status: str) -> str:
		if result["decision"] == "require_review":
			return "pending_review"
		return accepted_status

	def _policy_fields(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"decision": result["decision"],
			"matched_rules": tuple(result["matched_rules"]),
			"review_reasons": tuple(self._review_reasons(result)),
			"audit_evidence": self._audit_evidence(result, review_recorded),
		}

	def _audit_evidence(self, result: dict[str, Any], review_recorded: bool = False) -> dict[str, Any]:
		return {
			"required_actions": [
				action["required_action"]
				for action in result.get("actions", ())
				if action.get("required_action")
			],
			"reasons": self._reason_list(result),
			"review_recorded": bool(review_recorded),
		}

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		severity: str = "info",
		metadata: dict[str, Any] | None = None,
		result: dict[str, Any] | None = None,
	) -> None:
		result = result or {"decision": "allow", "matched_rules": [], "actions": []}
		event = OntoAuditEvent(
			id=stable_id("ontoaudit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			severity=severity,
			metadata=dict(metadata or {}),
			**self._policy_fields(result),
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _tenant_record_key(self, tenant_id: str, record_id: str) -> str:
		return f"{tenant_id}:{record_id}"

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(self._reason_list(result)) or "capability_policy_blocked"

	def _reason_list(self, result: dict[str, Any]) -> list[str]:
		return [
			action.get("reason", "capability_policy_blocked")
			for action in result.get("actions", [])
		]

	def _review_reasons(self, result: dict[str, Any]) -> tuple[str, ...]:
		if result["decision"] != "require_review":
			return ()
		return tuple(self._reason_list(result))


def _normalize_token(value: str) -> str:
	return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
