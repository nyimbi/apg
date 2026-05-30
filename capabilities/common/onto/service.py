"""Service layer for executable ontology management."""

from __future__ import annotations

from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract
from .models import (
	CurationReview,
	OntoAuditEvent,
	Ontology,
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
		self._audit_events: dict[str, OntoAuditEvent] = {}

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
		)
		self._ontologies[ontology.id] = ontology
		self._audit(tenant_id, "ontology_registered", ontology.id, f"Registered ontology {name}")
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
		)
		self._namespaces[namespace.id] = namespace
		self._touch_ontology(ontology)
		self._audit(tenant_id, "namespace_registered", namespace.id, f"Registered namespace {prefix}")
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
		self._raise_if_review_required(result)
		term = OntologyTerm(
			id=term_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			label=label,
			owner=owner,
			definition=definition,
			status=normalize_term_status(status),
			synonyms=list(synonyms or []),
			external_refs=list(external_refs or []),
			metadata=dict(metadata or {}),
		)
		self._terms[term.id] = term
		self._touch_ontology(ontology)
		self._audit(tenant_id, "term_created", term.id, f"Created term {label}")
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
		self._raise_if_review_required(result)
		review = CurationReview(
			id=review_id,
			tenant_id=tenant_id,
			ontology_id=term.ontology_id,
			subject_id=term.id,
			review_type=change_type,
			reviewer=reviewer,
			notes=notes,
		)
		self._reviews[review.id] = review
		term.status = normalize_term_status(status)
		term.updated_at = utc_now_iso()
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "term_curated", term.id, f"Curated term {term.label}")
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
		self._audit(tenant_id, "synonym_added", term.id, f"Added synonym {synonym}")
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
		)
		self._taxonomy_edges[edge.id] = edge
		self._touch_ontology(ontology)
		self._audit(tenant_id, "taxonomy_edge_added", edge.id, "Added taxonomy edge")
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
		self._raise_if_review_required(result)
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
			status="reviewed" if mapping_requires_review(normalized_confidence, self.confidence_threshold) else "active",
			metadata=dict(metadata or {}),
		)
		self._mappings[mapping.id] = mapping
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "semantic_mapping_created", mapping.id, f"Mapped term {term.label}")
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
		self._raise_if_review_required(result)
		review = CurationReview(
			id=review_id,
			tenant_id=tenant_id,
			ontology_id=term.ontology_id,
			subject_id=term.id,
			review_type="deprecation",
			reviewer=reviewer,
			notes=notes,
		)
		self._reviews[review.id] = review
		term.status = "deprecated"
		term.metadata["replacement_term_id"] = replacement_term_id
		term.updated_at = utc_now_iso()
		self._touch_ontology(self._ontologies[term.ontology_id])
		self._audit(tenant_id, "term_deprecated", term.id, f"Deprecated term {term.label}")
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
		self._raise_if_review_required(result)
		report = ValidationReport(
			id=report_id,
			tenant_id=tenant_id,
			ontology_id=ontology.id,
			issue_count=len(issues),
			issues=issues,
			status="passed" if ready else "issues_found",
			review_recorded=review_recorded,
			review_ref=review_ref,
		)
		self._validation_reports[report.id] = report
		self._audit(tenant_id, "ontology_validated", report.id, f"Validated ontology {ontology.name}")
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
		)
		self._reviews[review.id] = review
		mapping.review_recorded = True
		mapping.review_ref = review.id
		mapping.status = "reviewed"
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
		)
		self._publications[publication.id] = publication
		self._audit(tenant_id, "ontology_published", publication.id, f"Published ontology {ontology.name}")
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
		)
		self._exports[export.id] = export
		self._audit(tenant_id, "ontology_exported", export.id, f"Exported ontology {ontology.name}")
		return export.to_dict()

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

	def list_audit_events(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
		return self._list(self._audit_events, tenant_id)

	def dashboard_summary(self, tenant_id: str = "default") -> dict[str, Any]:
		self._require_tenant(tenant_id)
		terms = [item for item in self._terms.values() if item.tenant_id == tenant_id]
		mappings = [item for item in self._mappings.values() if item.tenant_id == tenant_id]
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
			"audit_event_count": len(self.list_audit_events(tenant_id)),
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

	def _raise_if_review_required(self, result: dict[str, Any]) -> None:
		self._raise_if_denied(result)
		if result["decision"] == "require_review":
			raise PermissionError(self._reasons(result))

	def _audit(
		self,
		tenant_id: str,
		event_type: str,
		subject_id: str,
		message: str,
		severity: str = "info",
		metadata: dict[str, Any] | None = None,
	) -> None:
		event = OntoAuditEvent(
			id=stable_id("ontoaudit", tenant_id, event_type, subject_id, len(self._audit_events)),
			tenant_id=tenant_id,
			event_type=event_type,
			subject_id=subject_id,
			message=message,
			severity=severity,
			metadata=dict(metadata or {}),
		)
		self._audit_events[event.id] = event

	def _list(self, records: dict[str, Any], tenant_id: str | None) -> list[dict[str, Any]]:
		values = list(records.values())
		if tenant_id is not None:
			values = [record for record in values if record.tenant_id == tenant_id]
		return [record.to_dict() for record in sorted(values, key=lambda item: item.id)]

	def _reasons(self, result: dict[str, Any]) -> str:
		return ", ".join(
			action.get("reason", "capability_policy_blocked")
			for action in result.get("actions", [])
		) or "capability_policy_blocked"
