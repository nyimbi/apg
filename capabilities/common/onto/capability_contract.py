"""Executable capability contract for APG Ontology Management."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_ONTO_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ONTO_AGENT_ROLES = [
	"namespace_steward",
	"term_curator",
	"taxonomy_reviewer",
	"mapping_reviewer",
	"validation_reviewer",
	"publication_reviewer",
	"import_reviewer",
	"vocabulary_steward",
	"lifecycle_batch_reviewer",
	"ontology_steward",
]
PRIVILEGED_ONTO_AGENT_ROLES = [
	"mapping_reviewer",
	"validation_reviewer",
	"publication_reviewer",
	"import_reviewer",
	"lifecycle_batch_reviewer",
	"ontology_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ontologies": {
		"id_required": True,
		"name_required": True,
		"owner_required": True,
		"domain_required": True,
		"versioning_enabled": True,
		"publication_approval_required": True,
		"retire_requires_review": True,
	},
	"namespaces": {
		"prefix_required": True,
		"uri_required": True,
		"owner_required": True,
		"unique_prefix_required": True,
	},
	"terms": {
		"label_required": True,
		"owner_required": True,
		"definition_required_for_publication": True,
		"allowed_statuses": ["draft", "curated", "published", "deprecated"],
		"deprecation_requires_replacement": True,
		"duplicate_detection_enabled": True,
		"synonym_management_enabled": True,
	},
	"taxonomy": {
		"cycle_detection_enabled": True,
		"self_relation_allowed": False,
		"allowed_relationships": ["broader_than", "narrower_than", "related_to", "equivalent_to"],
	},
	"mappings": {
		"confidence_threshold": 0.8,
		"target_required": True,
		"mapping_type_required": True,
		"external_mapping_review_required": True,
		"breaking_change_review_required": True,
		"allowed_types": ["exact", "close", "broad", "narrow", "related"],
	},
	"validation": {
		"required_before_publication": True,
		"duplicate_terms_block": True,
		"taxonomy_cycles_block": True,
		"draft_terms_block": True,
		"unreviewed_low_confidence_mappings_block": True,
	},
	"publication": {
		"approval_required": True,
		"validation_required": True,
		"auditable_version_required": True,
	},
	"import_export": {
		"allowed_import_formats": ["rdf", "owl", "jsonld", "skos", "csv"],
		"allowed_export_formats": ["rdf", "owl", "jsonld", "skos", "csv"],
		"large_import_review_threshold": 1000,
		"external_import_review_required": True,
	},
	"curation": {
		"curator_required": True,
		"evidence_required": True,
		"allowed_decisions": ["approved", "rejected", "needs_revision"],
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_ONTO_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ONTO_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_ONTO_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_ontology_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "onto.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"ontology_batch",
			"namespace_batch",
			"term_batch",
			"taxonomy_batch",
			"mapping_batch",
			"validation_batch",
			"publication_batch",
			"exchange_batch",
			"ontology_agent_batch",
		],
		"topics": [
			"onto.ontologies",
			"onto.namespaces",
			"onto.terms",
			"onto.taxonomy",
			"onto.mappings",
			"onto.validation",
			"onto.publication",
			"onto.exchange",
			"onto.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"security": {
		"cross_tenant_access_allowed": False,
		"rbac_required": True,
		"tenant_isolation_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_ontology_changes": True,
		"audit_term_changes": True,
		"audit_mapping_changes": True,
		"audit_publication": True,
		"curation_required": True,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.OntoService",
		"helper_runtime": "ontology_runtime.py",
		"http_api": "api.py",
		"event_stream": "bytewax",
		"knowledge_graph": "kngr",
		"metadata": "meta",
		"nlp": "nlpc",
		"graph": "grph",
		"search": "srch",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"cache": "cach",
		"metrics_sink": "moni",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_ontology_registry": True,
		"enable_namespace_manager": True,
		"enable_term_editor": True,
		"enable_taxonomy_editor": True,
		"enable_mapping_workbench": True,
		"enable_validation": True,
		"enable_imports": True,
		"enable_exports": True,
		"enable_publication_queue": True,
		"enable_governance": True,
		"enable_ontology_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "onto_vocabulary_workbench", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"ontologies",
		"namespaces",
		"terms",
		"taxonomy",
		"mappings",
		"validation",
		"publication",
		"import_export",
		"curation",
		"agents",
		"streaming",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"ontologies",
		"namespaces",
		"terms",
		"taxonomy",
		"mappings",
		"validation",
		"publication",
		"import_export",
		"curation",
		"agents",
		"streaming",
		"security",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ontology operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ontology_requires_id", "description": "Ontologies require stable identifiers.", "condition": {"operation": "register_ontology", "ontology_id_present": False}, "effect": {"decision": "deny", "reason": "ontology_id_required", "required_action": "attach_ontology_id"}},
	{"name": "ontology_requires_name", "description": "Ontologies require names.", "condition": {"operation": "register_ontology", "ontology_name_present": False}, "effect": {"decision": "deny", "reason": "ontology_name_required", "required_action": "attach_ontology_name"}},
	{"name": "ontology_requires_owner", "description": "Ontologies require accountable owners.", "condition": {"operation": "register_ontology", "ontology_owner_present": False}, "effect": {"decision": "deny", "reason": "ontology_owner_required", "required_action": "assign_owner"}},
	{"name": "ontology_requires_domain", "description": "Ontologies require domains.", "condition": {"operation": "register_ontology", "ontology_domain_present": False}, "effect": {"decision": "deny", "reason": "ontology_domain_required", "required_action": "attach_domain"}},
	{"name": "ontology_retire_requires_review", "description": "Retiring ontologies requires review.", "condition": {"operation": "retire_ontology", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "ontology_retire_review_required", "required_action": "record_ontology_review"}},
	{"name": "namespace_requires_ontology", "description": "Namespaces require an ontology.", "condition": {"operation": "register_namespace", "ontology_present": False}, "effect": {"decision": "deny", "reason": "ontology_required", "required_action": "select_ontology"}},
	{"name": "namespace_requires_prefix", "description": "Namespaces require prefixes.", "condition": {"operation": "register_namespace", "namespace_prefix_present": False}, "effect": {"decision": "deny", "reason": "namespace_prefix_required", "required_action": "attach_namespace_prefix"}},
	{"name": "namespace_requires_uri", "description": "Namespaces require URIs.", "condition": {"operation": "register_namespace", "namespace_uri_present": False}, "effect": {"decision": "deny", "reason": "namespace_uri_required", "required_action": "attach_namespace_uri"}},
	{"name": "namespace_requires_owner", "description": "Namespaces require owners.", "condition": {"operation": "register_namespace", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "namespace_owner_required", "required_action": "assign_namespace_owner"}},
	{"name": "namespace_prefix_must_be_unique", "description": "Namespace prefixes must be unique per ontology.", "condition": {"operation": "register_namespace", "namespace_prefix_unique": False}, "effect": {"decision": "deny", "reason": "namespace_prefix_duplicate", "required_action": "choose_unique_prefix"}},
	{"name": "term_requires_ontology", "description": "Ontology terms require an ontology.", "condition": {"operation": "create_term", "ontology_present": False}, "effect": {"decision": "deny", "reason": "ontology_required", "required_action": "select_ontology"}},
	{"name": "term_requires_label", "description": "Ontology terms require labels.", "condition": {"operation": "create_term", "term_label_present": False}, "effect": {"decision": "deny", "reason": "term_label_required", "required_action": "attach_term_label"}},
	{"name": "term_requires_owner", "description": "Ontology terms require an owner.", "condition": {"operation": "create_term", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "term_owner_required", "required_action": "assign_term_owner"}},
	{"name": "term_status_requires_allowed_value", "description": "Term status must be a configured value.", "condition": {"operation": "create_term", "term_status_allowed": False}, "effect": {"decision": "deny", "reason": "term_status_invalid", "required_action": "choose_allowed_status"}},
	{"name": "duplicate_term_requires_resolution", "description": "Duplicate term labels require resolution.", "condition": {"operation": "create_term", "duplicate_term_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "duplicate_term_review_required", "required_action": "record_duplicate_resolution"}},
	{"name": "term_deprecation_requires_replacement", "description": "Deprecated terms require a replacement term.", "condition": {"operation": "deprecate_term", "replacement_term_present": False}, "effect": {"decision": "deny", "reason": "replacement_term_required", "required_action": "select_replacement_term"}},
	{"name": "term_deprecation_requires_review", "description": "Term deprecation requires review.", "condition": {"operation": "deprecate_term", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "term_deprecation_review_required", "required_action": "record_deprecation_review"}},
	{"name": "synonym_requires_term", "description": "Synonyms require existing terms.", "condition": {"operation": "add_synonym", "term_present": False}, "effect": {"decision": "deny", "reason": "term_required", "required_action": "select_term"}},
	{"name": "synonym_requires_value", "description": "Synonyms require a value.", "condition": {"operation": "add_synonym", "synonym_present": False}, "effect": {"decision": "deny", "reason": "synonym_required", "required_action": "attach_synonym"}},
	{"name": "taxonomy_requires_parent", "description": "Taxonomy edges require parent terms.", "condition": {"operation": "add_taxonomy_edge", "parent_term_present": False}, "effect": {"decision": "deny", "reason": "parent_term_required", "required_action": "select_parent_term"}},
	{"name": "taxonomy_requires_child", "description": "Taxonomy edges require child terms.", "condition": {"operation": "add_taxonomy_edge", "child_term_present": False}, "effect": {"decision": "deny", "reason": "child_term_required", "required_action": "select_child_term"}},
	{"name": "taxonomy_self_relation_denied", "description": "Taxonomy self-relations are denied.", "condition": {"operation": "add_taxonomy_edge", "self_relation": True}, "effect": {"decision": "deny", "reason": "taxonomy_self_relation_denied", "required_action": "choose_distinct_terms"}},
	{"name": "taxonomy_cycle_blocks_edge", "description": "Taxonomy cycles block edge creation.", "condition": {"operation": "add_taxonomy_edge", "taxonomy_cycle_detected": True}, "effect": {"decision": "deny", "reason": "taxonomy_cycle_detected", "required_action": "remove_cycle"}},
	{"name": "taxonomy_relationship_requires_allowed_value", "description": "Taxonomy relationship types must be configured values.", "condition": {"operation": "add_taxonomy_edge", "relationship_type_allowed": False}, "effect": {"decision": "deny", "reason": "taxonomy_relationship_invalid", "required_action": "choose_allowed_relationship"}},
	{"name": "mapping_requires_term", "description": "Semantic mappings require terms.", "condition": {"operation": "create_mapping", "term_present": False}, "effect": {"decision": "deny", "reason": "term_required", "required_action": "select_term"}},
	{"name": "mapping_requires_target", "description": "Semantic mappings require target references.", "condition": {"operation": "create_mapping", "target_ref_present": False}, "effect": {"decision": "deny", "reason": "mapping_target_required", "required_action": "attach_target_ref"}},
	{"name": "mapping_type_requires_allowed_value", "description": "Mapping types must be configured values.", "condition": {"operation": "create_mapping", "mapping_type_allowed": False}, "effect": {"decision": "deny", "reason": "mapping_type_invalid", "required_action": "choose_allowed_mapping_type"}},
	{"name": "low_confidence_mapping_requires_review", "description": "Low-confidence semantic mappings require review.", "condition": {"operation": "create_mapping", "mapping_confidence_lt": 0.8, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "mapping_review_required", "required_action": "record_mapping_review"}},
	{"name": "external_mapping_requires_review", "description": "External semantic mappings require review.", "condition": {"operation": "create_mapping", "mapping_scope": "external", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_mapping_review_required", "required_action": "record_mapping_review"}},
	{"name": "breaking_change_requires_review", "description": "Breaking ontology changes require review.", "condition": {"change_type": "breaking", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "breaking_change_review_required", "required_action": "record_breaking_change_review"}},
	{"name": "curation_requires_reviewer", "description": "Curation requires a reviewer.", "condition": {"operation": "curate_term", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "reviewer_required", "required_action": "assign_reviewer"}},
	{"name": "curation_requires_evidence", "description": "Curation requires evidence for non-draft status changes.", "condition": {"operation": "curate_term", "curation_evidence_present": False}, "effect": {"decision": "deny", "reason": "curation_evidence_required", "required_action": "attach_curation_evidence"}},
	{"name": "validation_requires_ontology", "description": "Validation requires an ontology.", "condition": {"operation": "validate_ontology", "ontology_present": False}, "effect": {"decision": "deny", "reason": "ontology_required", "required_action": "select_ontology"}},
	{"name": "validation_issue_count_requires_review", "description": "Validation issues require review before publication.", "condition": {"operation": "validate_ontology", "issue_count_gt": 0, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "validation_review_required", "required_action": "record_validation_review"}},
	{"name": "publication_requires_approval", "description": "Ontology publication requires approval.", "condition": {"operation": "publish_ontology", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "record_publication_approval"}},
	{"name": "publication_requires_validation", "description": "Ontology publication requires a validation report.", "condition": {"operation": "publish_ontology", "validation_recorded": False}, "effect": {"decision": "deny", "reason": "validation_required", "required_action": "validate_ontology"}},
	{"name": "duplicate_term_blocks_publication", "description": "Duplicate terms block ontology publication.", "condition": {"operation": "publish_ontology", "duplicate_term_detected": True}, "effect": {"decision": "deny", "reason": "duplicate_term_detected", "required_action": "resolve_duplicate_term"}},
	{"name": "taxonomy_cycle_blocks_publication", "description": "Taxonomy cycles block ontology publication.", "condition": {"operation": "publish_ontology", "taxonomy_cycle_detected": True}, "effect": {"decision": "deny", "reason": "taxonomy_cycle_detected", "required_action": "resolve_taxonomy_cycle"}},
	{"name": "draft_terms_block_publication", "description": "Draft terms block ontology publication.", "condition": {"operation": "publish_ontology", "draft_terms_present": True}, "effect": {"decision": "deny", "reason": "draft_terms_present", "required_action": "curate_terms"}},
	{"name": "unreviewed_mappings_block_publication", "description": "Unreviewed low-confidence mappings block publication.", "condition": {"operation": "publish_ontology", "unreviewed_low_confidence_mappings_present": True}, "effect": {"decision": "deny", "reason": "mapping_review_required", "required_action": "review_low_confidence_mappings"}},
	{"name": "large_import_requires_review", "description": "Large ontology imports require review.", "condition": {"operation": "import_ontology", "record_count_gt": 1000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_import_review_required", "required_action": "record_import_review"}},
	{"name": "external_import_requires_review", "description": "External ontology imports require review.", "condition": {"operation": "import_ontology", "source_scope": "external", "review_recorded": False}, "effect": {"decision": "require_review", "reason": "external_import_review_required", "required_action": "record_import_review"}},
	{"name": "export_requires_allowed_format", "description": "Ontology exports require configured formats.", "condition": {"operation": "export_ontology", "export_format_allowed": False}, "effect": {"decision": "deny", "reason": "export_format_invalid", "required_action": "choose_allowed_export_format"}},
	{"name": "batch_ontology_mutation_requires_bytewax", "description": "Batch ontology mutations must use Bytewax event streams.", "condition": {"operation": "batch_ontology_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_ontology_access_denied", "description": "Ontology operations may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_ontology_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "ontology_state_change_requires_audit", "description": "Ontology state changes require audit events.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "audit_event_required", "required_action": "record_audit_event"}},
	{"name": "ontology_agent_runtime_supported", "description": "Ontology agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_ontology_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ontology_agent_runtime", "required_action": "choose_supported_ontology_agent_runtime"}},
	{"name": "ontology_agent_role_supported", "description": "Ontology agents must use supported curation, mapping, validation, publication, or lifecycle roles.", "condition": {"operation": "register_ontology_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ontology_agent_role", "required_action": "choose_supported_ontology_agent_role"}},
	{"name": "ontology_agent_requires_scope", "description": "Ontology agents require an explicit bounded ontology, namespace, term, taxonomy, mapping, publication, or exchange scope.", "condition": {"operation": "register_ontology_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "ontology_agent_scope_required", "required_action": "declare_ontology_agent_scope"}},
	{"name": "ontology_agent_requires_owner", "description": "Ontology agents require an accountable owner.", "condition": {"operation": "register_ontology_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "ontology_agent_owner_required", "required_action": "assign_ontology_agent_owner"}},
	{"name": "ontology_agent_requires_purpose", "description": "Ontology agents require a documented purpose.", "condition": {"operation": "register_ontology_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "ontology_agent_purpose_required", "required_action": "document_ontology_agent_purpose"}},
	{"name": "ontology_agent_requires_contribution_disclosure", "description": "Ontology agents must disclose machine-authored curation, mapping, validation, and publication contributions.", "condition": {"operation": "register_ontology_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ontology_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "ontology_agent_privileged_role_requires_human_approval", "description": "Privileged ontology-agent roles require human approval evidence.", "condition": {"operation": "register_ontology_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "ontology_agent_human_approval_required", "required_action": "record_human_ontology_agent_approval"}},
	{"name": "bytewax_ontology_stream_required", "description": "ONTO lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_onto_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_onto_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/onto/dashboard", "component": "ONTODashboard", "permission": "onto:view", "nav_group": "Overview"},
	{"name": "ontologies", "path": "/onto/ontologies", "component": "OntologyRegistry", "permission": "onto:view", "nav_group": "Registry"},
	{"name": "namespaces", "path": "/onto/namespaces", "component": "NamespaceManager", "permission": "onto:edit", "nav_group": "Registry"},
	{"name": "terms", "path": "/onto/terms", "component": "TermEditor", "permission": "onto:edit", "nav_group": "Vocabulary"},
	{"name": "taxonomy", "path": "/onto/taxonomy", "component": "TaxonomyEditor", "permission": "onto:edit", "nav_group": "Vocabulary"},
	{"name": "mappings", "path": "/onto/mappings", "component": "MappingWorkbench", "permission": "onto:map", "nav_group": "Mappings"},
	{"name": "validation", "path": "/onto/validation", "component": "OntologyValidation", "permission": "onto:govern", "nav_group": "Governance"},
	{"name": "imports", "path": "/onto/imports", "component": "OntologyImportWorkbench", "permission": "onto:edit", "nav_group": "Exchange"},
	{"name": "exports", "path": "/onto/exports", "component": "OntologyExportWorkbench", "permission": "onto:view", "nav_group": "Exchange"},
	{"name": "publication", "path": "/onto/publication", "component": "PublicationQueue", "permission": "onto:publish", "nav_group": "Governance"},
	{"name": "governance", "path": "/onto/governance", "component": "OntologyGovernance", "permission": "onto:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/onto/agents", "component": "OntologyAgentRoster", "permission": "onto:govern", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/onto/lifecycle", "component": "ONTOLifecycleBatchMonitor", "permission": "onto:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/onto/audit", "component": "OntologyAuditTimeline", "permission": "onto:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/onto/settings", "component": "ONTOSettings", "permission": "onto:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "onto_vocabulary_workbench",
	"tokens": {
		"color.primary": "#4B5563",
		"color.accent": "#F2A541",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"ontology_card": {"icon": "book-open", "status_indicator": "version-pill", "risk_style": "publication-band"},
		"namespace_panel": {"visual": "prefix-list", "status_style": "uri-chip"},
		"term_card": {"icon": "book-a", "status_indicator": "term-status-pill", "risk_style": "definition-band"},
		"taxonomy_tree": {"visual": "hierarchy-tree", "highlight": "selected-term-chip"},
		"mapping_panel": {"visual": "concept-map", "threshold_style": "confidence-band"},
		"validation_report": {"visual": "issue-list", "status_style": "readiness-chip"},
		"publication_queue": {"visual": "approval-list", "status_style": "review-chip"},
		"exchange_panel": {"visual": "import-export-list", "status_style": "format-chip"},
		"ontology_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "governance-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class ONTO agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_ONTO_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_ONTO_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_ONTO_AGENT_ROLES),
		"required_fields": ["tenant_id", "agent_id", "name", "runtime", "role", "scope", "owner", "purpose"],
		"guardrails": [
			"supported_runtime",
			"supported_role",
			"explicit_scope",
			"accountable_owner",
			"declared_purpose",
			"machine_contribution_disclosure",
			"human_approval_for_privileged_roles",
		],
		"adapter_contract": "aicr_provider_neutral_ontology_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the ONTO Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "onto.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"ontology_batch",
			"namespace_batch",
			"term_batch",
			"taxonomy_batch",
			"mapping_batch",
			"validation_batch",
			"publication_batch",
			"exchange_batch",
			"ontology_agent_batch",
		],
		"topics": [
			"onto.ontologies",
			"onto.namespaces",
			"onto.terms",
			"onto.taxonomy",
			"onto.mappings",
			"onto.validation",
			"onto.publication",
			"onto.exchange",
			"onto.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "onto",
		"display_name": "Ontology Management",
		"provides": ["ontology_management", "semantic_vocabulary_governance", "ontology_agent_composition"],
		"requires": ["kngr", "meta", "nlpc", "grph", "srch", "aicr", "conf", "auth", "audl"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/onto/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	return _evaluate(RULES, context)


def _evaluate(rules: list[dict[str, Any]], context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in rules:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
