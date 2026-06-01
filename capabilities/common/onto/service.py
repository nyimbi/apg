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
		self._lifecycle_operations = set(contract["streaming"]["required_operations"])

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
