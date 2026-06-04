"""Executable capability contract for APG Data Loss Prevention."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_DLPD_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_DLPD_AGENT_ROLES: list[str] = [
	"policy_reviewer",
	"classifier_reviewer",
	"inspection_triage_agent",
	"quarantine_reviewer",
	"incident_response_reviewer",
	"privacy_reviewer",
	"legal_hold_reviewer",
	"lifecycle_batch_reviewer",
	"dlp_steward",
]

PRIVILEGED_DLPD_AGENT_ROLES: list[str] = [
	"quarantine_reviewer",
	"incident_response_reviewer",
	"privacy_reviewer",
	"legal_hold_reviewer",
	"lifecycle_batch_reviewer",
	"dlp_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"data_patterns": {
		"enabled_classifiers": ["pii", "phi", "pci", "secrets", "financial_records", "source_code"],
		"nlp_classification_enabled": True,
		"minimum_classifier_confidence": 0.82,
		"custom_pattern_review_required": True,
		"sensitive_label_required": True,
	},
	"policies": {
		"owner_required": True,
		"active_policy_required": True,
		"channels_required": True,
		"classifiers_required": True,
		"default_actions": ["allow", "alert", "block", "quarantine"],
	},
	"channels": {
		"inspected": ["email", "api_export", "file_share", "chat", "clipboard", "object_storage"],
		"egress_policy_required": True,
		"bulk_export_threshold_records": 10000,
		"anomaly_context_required": True,
		"destination_required": True,
	},
	"classification": {
		"sensitive_content_requires_label": True,
		"high_confidence_threshold": 0.9,
		"secret_patterns_high_severity": True,
		"source_code_review_required": True,
	},
	"response": {
		"block_high_severity": True,
		"quarantine_supported": True,
		"incident_owner_required": True,
		"notification_required": True,
		"legal_hold_supported": True,
	},
	"quarantine": {
		"encrypted_required": True,
		"legal_hold_default": True,
		"release_review_required": True,
		"content_hash_required": True,
	},
	"incidents": {
		"owner_required": True,
		"resolution_required": True,
		"duplicate_open_incident_blocked": True,
		"severity_required": True,
	},
	"reviews": {
		"large_export_review_required": True,
		"independent_reviewer_required": True,
		"review_notes_required": True,
		"sensitive_destination_review_required": True,
	},
	"security": {
		"tenant_isolation_required": True,
		"encrypted_quarantine_required": True,
		"raw_content_retention_allowed": False,
		"policy_mutation_audit_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_inspection": True,
		"encrypted_quarantine_required": True,
		"legal_hold_supported": True,
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_DLPD_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_DLPD_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_DLPD_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_dlp_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "dlpd.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"policy_batch",
			"classifier_batch",
			"inspection_batch",
			"quarantine_batch",
			"incident_batch",
			"review_batch",
			"dlp_agent_batch",
		],
		"topics": [
			"dlpd.policies",
			"dlpd.classifiers",
			"dlpd.inspections",
			"dlpd.quarantine",
			"dlpd.incidents",
			"dlpd.reviews",
			"dlpd.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "service.DlpdService",
		"helper_runtime": "dlp_engine.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"security_framework": "secu",
		"encryption": "encr",
		"nlp_core": "nlpc",
		"anomaly_detection": "anom",
		"audit_sink": "audl",
		"message_bus": "mqeb",
		"search": "srch",
		"compliance": "comp",
		"monitoring": "moni",
		"cache": "cach",
		"agent_adapter": "aicr_provider_neutral_dlp_agent_adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_policy_console": True,
		"enable_classifier_workbench": True,
		"enable_channel_monitor": True,
		"enable_inspection_workbench": True,
		"enable_incident_queue": True,
		"enable_quarantine_vault": True,
		"enable_review_queue": True,
		"enable_legal_hold": True,
		"enable_analytics": True,
		"enable_dlp_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "dlpd_data_protection_ops", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"data_patterns",
		"policies",
		"channels",
		"classification",
		"response",
		"quarantine",
		"incidents",
		"reviews",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"data_patterns",
		"policies",
		"channels",
		"classification",
		"response",
		"quarantine",
		"incidents",
		"reviews",
		"security",
		"governance",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All DLP operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "policy_requires_owner", "description": "DLP policies require an accountable owner.", "condition": {"operation": "register_policy", "owner_present": False}, "effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_policy_owner"}},
	{"name": "policy_requires_channels", "description": "DLP policies require at least one inspected channel.", "condition": {"operation": "register_policy", "channels_present": False}, "effect": {"decision": "deny", "reason": "policy_channels_required", "required_action": "select_channels"}},
	{"name": "policy_requires_classifiers", "description": "DLP policies require at least one classifier.", "condition": {"operation": "register_policy", "classifiers_present": False}, "effect": {"decision": "require_review", "reason": "policy_classifiers_required", "required_action": "attach_classifiers"}},
	{"name": "policy_requires_egress_binding", "description": "DLP policies require egress policy binding.", "condition": {"operation": "register_policy", "egress_policy_attached": False}, "effect": {"decision": "deny", "reason": "egress_policy_required", "required_action": "attach_egress_policy"}},
	{"name": "inspection_source_requires_policy", "description": "Inspected egress sources require a policy.", "condition": {"operation": "inspect_egress", "egress_policy_attached": False}, "effect": {"decision": "deny", "reason": "egress_policy_required", "required_action": "attach_egress_policy"}},
	{"name": "inspection_requires_active_policy", "description": "Inspections require an active policy.", "condition": {"operation": "inspect_egress", "policy_active": False}, "effect": {"decision": "deny", "reason": "active_dlp_policy_required", "required_action": "activate_policy"}},
	{"name": "inspection_requires_covered_channel", "description": "Inspections must use channels covered by the selected policy.", "condition": {"operation": "inspect_egress", "channel_covered": False}, "effect": {"decision": "deny", "reason": "channel_not_covered_by_policy", "required_action": "select_covered_channel"}},
	{"name": "inspection_requires_destination", "description": "Inspections require a destination.", "condition": {"operation": "inspect_egress", "destination_present": False}, "effect": {"decision": "deny", "reason": "destination_required", "required_action": "record_destination"}},
	{"name": "classifier_requires_label", "description": "Classifiers require a sensitivity label.", "condition": {"operation": "register_classifier", "sensitivity_label_present": False}, "effect": {"decision": "deny", "reason": "sensitivity_label_required", "required_action": "set_sensitivity_label"}},
	{"name": "classifier_requires_patterns", "description": "Classifiers require one or more pattern keys.", "condition": {"operation": "register_classifier", "pattern_keys_present": False}, "effect": {"decision": "deny", "reason": "classifier_patterns_required", "required_action": "attach_classifier_patterns"}},
	{"name": "custom_classifier_requires_review", "description": "Custom DLP classifiers require review.", "condition": {"operation": "register_classifier", "classifier_type": "custom", "classifier_review_recorded": False}, "effect": {"decision": "deny", "reason": "custom_pattern_review_required", "required_action": "review_custom_classifier"}},
	{"name": "classifier_confidence_requires_threshold", "description": "Classifier confidence must meet the tenant threshold.", "condition": {"operation": "classify_content", "classifier_confidence_lt": 0.82}, "effect": {"decision": "require_review", "reason": "classifier_confidence_review_required", "required_action": "review_classifier_hit"}},
	{"name": "sensitive_content_requires_classification", "description": "Sensitive content cannot move without classification metadata.", "condition": {"sensitive_content_detected": True, "classification_label_present": False}, "effect": {"decision": "deny", "reason": "classification_label_required", "required_action": "apply_classification_label"}},
	{"name": "source_code_requires_review", "description": "Source-code egress requires review.", "condition": {"source_code_detected": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "source_code_review_required", "required_action": "review_source_code_egress"}},
	{"name": "secret_exfiltration_requires_block", "description": "Secret exfiltration must be blocked or quarantined.", "condition": {"secret_detected": True, "blocked_or_quarantined": False}, "effect": {"decision": "deny", "reason": "secret_exfiltration_block_required", "required_action": "block_or_quarantine_transfer"}},
	{"name": "high_severity_exfiltration_requires_block", "description": "High-severity exfiltration signals must be blocked or quarantined.", "condition": {"severity": "high", "blocked_or_quarantined": False}, "effect": {"decision": "deny", "reason": "high_severity_block_required", "required_action": "block_or_quarantine_transfer"}},
	{"name": "medium_severity_requires_alert_or_quarantine", "description": "Medium-severity egress requires alert, block, or quarantine.", "condition": {"severity": "medium", "alerted_or_quarantined": False}, "effect": {"decision": "require_review", "reason": "medium_severity_response_required", "required_action": "alert_or_quarantine_transfer"}},
	{"name": "large_export_requires_review", "description": "Large exports require review before release.", "condition": {"export_record_count_gt": 10000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_export_review_required", "required_action": "review_export"}},
	{"name": "external_destination_requires_policy", "description": "External destinations require an egress policy.", "condition": {"external_destination": True, "egress_policy_attached": False}, "effect": {"decision": "deny", "reason": "external_destination_policy_required", "required_action": "attach_external_destination_policy"}},
	{"name": "restricted_destination_requires_review", "description": "Restricted destinations require review.", "condition": {"restricted_destination": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "restricted_destination_review_required", "required_action": "review_destination"}},
	{"name": "quarantine_requires_encryption", "description": "Quarantined sensitive data must be encrypted.", "condition": {"quarantine_requested": True, "quarantine_encrypted": False}, "effect": {"decision": "deny", "reason": "encrypted_quarantine_required", "required_action": "encrypt_quarantine"}},
	{"name": "quarantine_requires_content_hash", "description": "Quarantine entries require content hashes.", "condition": {"operation": "create_quarantine_item", "content_hash_present": False}, "effect": {"decision": "deny", "reason": "quarantine_content_hash_required", "required_action": "record_content_hash"}},
	{"name": "quarantine_release_requires_review", "description": "Quarantine release requires review.", "condition": {"operation": "release_quarantine", "release_review_recorded": False}, "effect": {"decision": "deny", "reason": "quarantine_release_review_required", "required_action": "review_quarantine_release"}},
	{"name": "legal_hold_release_blocked", "description": "Items on legal hold cannot be released.", "condition": {"operation": "release_quarantine", "legal_hold_active": True}, "effect": {"decision": "deny", "reason": "legal_hold_release_blocked", "required_action": "clear_legal_hold"}},
	{"name": "incident_requires_owner", "description": "DLP incidents require an owner.", "condition": {"operation": "open_incident", "owner_present": False}, "effect": {"decision": "deny", "reason": "incident_owner_required", "required_action": "assign_incident_owner"}},
	{"name": "incident_requires_severity", "description": "DLP incidents require severity.", "condition": {"operation": "open_incident", "severity_present": False}, "effect": {"decision": "deny", "reason": "incident_severity_required", "required_action": "record_incident_severity"}},
	{"name": "incident_resolution_requires_notes", "description": "DLP incident resolution requires notes.", "condition": {"operation": "resolve_incident", "resolution_present": False}, "effect": {"decision": "deny", "reason": "incident_resolution_required", "required_action": "record_resolution"}},
	{"name": "duplicate_open_incident_blocked", "description": "Duplicate open incidents are blocked.", "condition": {"operation": "open_incident", "duplicate_open_incident": True}, "effect": {"decision": "deny", "reason": "duplicate_open_incident", "required_action": "resolve_existing_incident"}},
	{"name": "notification_required_for_incident", "description": "Incidents require notification evidence.", "condition": {"operation": "open_incident", "notification_sent": False}, "effect": {"decision": "require_review", "reason": "incident_notification_required", "required_action": "send_incident_notification"}},
	{"name": "review_requires_independent_reviewer", "description": "DLP reviews require an independent reviewer.", "condition": {"operation": "review_export", "reviewer_same_as_subject": True}, "effect": {"decision": "deny", "reason": "independent_dlp_review_required", "required_action": "route_to_independent_reviewer"}},
	{"name": "review_decision_requires_notes", "description": "DLP review decisions require notes.", "condition": {"operation": "review_export", "notes_present": False}, "effect": {"decision": "require_review", "reason": "dlp_review_notes_required", "required_action": "record_review_notes"}},
	{"name": "raw_content_retention_denied", "description": "The generated runtime may not retain raw sensitive content.", "condition": {"raw_content_retention_requested": True}, "effect": {"decision": "deny", "reason": "raw_content_retention_denied", "required_action": "store_hash_and_metadata_only"}},
	{"name": "batch_dlp_mutation_requires_bytewax", "description": "Batch DLP mutations must use Bytewax event streams.", "condition": {"operation": "batch_dlp_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_dlp_access_denied", "description": "DLP records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_dlp_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "dlp_state_change_requires_audit", "description": "DLP state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "dlp_audit_event_required", "required_action": "record_dlp_audit_event"}},
	{"name": "dlp_agent_runtime_supported", "description": "DLP agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_dlp_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_dlp_agent_runtime", "required_action": "choose_supported_dlpd_agent_runtime"}},
	{"name": "dlp_agent_role_supported", "description": "DLP agents must use supported data-protection roles.", "condition": {"operation": "register_dlp_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_dlp_agent_role", "required_action": "choose_supported_dlpd_agent_role"}},
	{"name": "dlp_agent_requires_scope", "description": "DLP agents require an explicit policy, classifier, channel, inspection, quarantine, incident, legal-hold, or lifecycle scope.", "condition": {"operation": "register_dlp_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "dlp_agent_scope_required", "required_action": "declare_dlp_agent_scope"}},
	{"name": "dlp_agent_requires_owner", "description": "DLP agents require an accountable owner.", "condition": {"operation": "register_dlp_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "dlp_agent_owner_required", "required_action": "assign_dlp_agent_owner"}},
	{"name": "dlp_agent_requires_purpose", "description": "DLP agents require a documented data-protection purpose.", "condition": {"operation": "register_dlp_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "dlp_agent_purpose_required", "required_action": "document_dlp_agent_purpose"}},
	{"name": "dlp_agent_requires_contribution_disclosure", "description": "DLP agents must disclose machine-authored policy, classifier, inspection, quarantine, incident, legal-hold, and lifecycle-review contributions.", "condition": {"operation": "register_dlp_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "dlp_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "dlp_agent_privileged_role_requires_human_approval", "description": "Privileged DLP agent roles require human approval evidence.", "condition": {"operation": "register_dlp_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "dlp_agent_human_approval_required", "required_action": "record_human_dlp_agent_approval"}},
	{"name": "dlpd_lifecycle_batch_requires_mutations", "description": "DLPD lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_dlpd_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "dlpd_lifecycle_batch_empty", "required_action": "include_dlpd_lifecycle_mutations"}},
	{"name": "bytewax_dlpd_stream_required", "description": "DLPD lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_dlpd_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_dlpd_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dlpd/dashboard", "component": "DLPDDashboard", "permission": "dlpd:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/dlpd/policies", "component": "DLPPolicyConsole", "permission": "dlpd:manage_policies", "nav_group": "Policies"},
	{"name": "classifiers", "path": "/dlpd/classifiers", "component": "ClassifierWorkbench", "permission": "dlpd:manage_policies", "nav_group": "Policies"},
	{"name": "channels", "path": "/dlpd/channels", "component": "ChannelMonitor", "permission": "dlpd:inspect", "nav_group": "Monitoring"},
	{"name": "inspections", "path": "/dlpd/inspections", "component": "InspectionWorkbench", "permission": "dlpd:inspect", "nav_group": "Monitoring"},
	{"name": "incidents", "path": "/dlpd/incidents", "component": "IncidentQueue", "permission": "dlpd:respond", "nav_group": "Response"},
	{"name": "quarantine", "path": "/dlpd/quarantine", "component": "QuarantineVault", "permission": "dlpd:respond", "nav_group": "Response"},
	{"name": "reviews", "path": "/dlpd/reviews", "component": "DLPReviewQueue", "permission": "dlpd:review", "nav_group": "Response"},
	{"name": "legal_hold", "path": "/dlpd/legal-hold", "component": "LegalHoldConsole", "permission": "dlpd:respond", "nav_group": "Governance"},
	{"name": "analytics", "path": "/dlpd/analytics", "component": "DLPAnalytics", "permission": "dlpd:view", "nav_group": "Operations"},
	{"name": "agents", "path": "/dlpd/agents", "component": "DLPAgentRoster", "permission": "dlpd:admin", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/dlpd/lifecycle", "component": "DLPDLifecycleBatchMonitor", "permission": "dlpd:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/dlpd/audit", "component": "DLPAuditTrail", "permission": "dlpd:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/dlpd/settings", "component": "DLPDSettings", "permission": "dlpd:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "dlpd_data_protection_ops",
	"tokens": {
		"color.primary": "#254E58",
		"color.accent": "#B83280",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F9FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"classifier_grid": {"icon": "scan-text", "status_indicator": "classifier-pill", "risk_style": "sensitivity-band"},
		"policy_matrix": {"visual": "policy-channel-grid", "status_style": "policy-chip"},
		"channel_flow": {"visual": "egress-sankey", "highlight": "blocked-chip"},
		"inspection_table": {"visual": "egress-table", "status_style": "decision-chip"},
		"incident_queue": {"visual": "severity-lanes", "status_style": "response-chip"},
		"quarantine_vault": {"visual": "encrypted-item-list", "status_style": "hold-chip"},
		"review_queue": {"visual": "decision-lane", "status_style": "review-chip"},
		"legal_hold": {"visual": "hold-ledger", "status_style": "hold-chip"},
		"dlp_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "data-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "digest-chip"},
	},
}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.dlpd.lifecycle",
	"key": "tenant_id",
	"events": [
		"policy_created",
		"policy_updated",
		"policy_activated",
		"policy_deactivated",
		"scan_completed",
		"violation_detected",
		"violation_remediated",
		"incident_raised",
		"incident_resolved",
		"quarantine_applied",
		"classification_updated",
		"agent_registered",
	],
	"guardrails": [
		"dlpd_batch_requires_bytewax",
		"dlpd_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable DLPD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "dlpd",
		"display_name": "Data Loss Prevention",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"agents": agent_manifest(),
		"streaming": deepcopy(STREAMING),
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/dlpd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def agent_manifest() -> dict[str, Any]:
	"""Return provider-neutral DLP agent composition metadata."""
	agents = DEFAULT_CONFIGURATION["agents"]
	return {
		"first_class": bool(agents["first_class"]),
		"supported_runtimes": list(agents["supported_runtimes"]),
		"supported_roles": list(agents["supported_roles"]),
		"privileged_roles": list(agents["privileged_roles"]),
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
		"adapter_contract": agents["adapter_contract"],
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return Bytewax lifecycle stream metadata for DLPD composition."""
	streaming = DEFAULT_CONFIGURATION["streaming"]
	return {
		"engine": streaming["engine"],
		"lifecycle_stream": streaming["lifecycle_stream"],
		"watermark": streaming["watermark"],
		"required_processor": streaming["required_processor"],
		"required_operations": list(streaming["required_operations"]),
		"topics": list(streaming["topics"]),
		"broker_core_dependency_allowed": bool(streaming["broker_core_dependency_allowed"]),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default DLPD governance rules."""
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
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
