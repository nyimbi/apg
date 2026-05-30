"""Executable capability contract for APG Digital Forms and eSign."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"forms": {
		"template_owner_required": True,
		"schema_validation_required": True,
		"publication_approval_required": True,
		"regulated_field_dlp_required": True,
		"form_state_audit_required": True,
	},
	"submissions": {
		"schema_validation_required": True,
		"audit_trail_required": True,
		"validation_hash_required": True,
	},
	"envelopes": {
		"subject_required": True,
		"recipient_required": True,
		"document_hash_required": True,
		"expiry_required": True,
		"expiry_in_future_required": True,
		"routing_order_required": True,
		"ordered_signing_required": True,
		"state_change_audit_required": True,
	},
	"signatures": {
		"identity_verification_required": True,
		"signature_intent_required": True,
		"tamper_seal_required": True,
		"multi_party_routing_enabled": True,
		"signer_consent_required": True,
		"sign_after_completion_blocked": True,
		"duplicate_signing_blocked": True,
	},
	"evidence": {
		"audit_trail_required": True,
		"encrypted_evidence_required": True,
		"certificate_of_completion": True,
		"retention_policy_required": True,
		"tamper_seal_verification_required": True,
	},
	"signing_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": ["codex", "claude_code", "opencode", "pi"],
		"allowed_roles": ["form_assistant", "clause_reviewer", "routing_coordinator", "evidence_auditor"],
	},
	"governance": {
		"require_tenant_context": True,
		"compliance_framework_link_required": True,
		"recipient_consent_required": True,
		"delegated_signing_policy_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"envelope_metrics_required": True,
		"evidence_metrics_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.EsgnService",
		"sealing_engine": "signing_engine.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"workflow": "wflo",
		"identity": "auth",
		"encryption": "encr",
		"audit_sink": "audl",
		"compliance": "comp",
		"notifications": "ntfy",
		"document_intelligence": "nlpc",
		"theme": "them",
	},
	"ui": {
		"enable_form_builder": True,
		"enable_envelope_console": True,
		"enable_signing_room": True,
		"enable_evidence_vault": True,
		"enable_agent_panel": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "esgn_forms_signing",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"forms",
		"submissions",
		"envelopes",
		"signatures",
		"evidence",
		"signing_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"forms",
		"submissions",
		"envelopes",
		"signatures",
		"evidence",
		"signing_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1}
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All e-sign operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "form_template_requires_owner", "description": "Form templates require an accountable owner.", "condition": {"operation": "create_form_template", "template_owner_assigned": False}, "effect": {"decision": "deny", "reason": "template_owner_required", "required_action": "assign_template_owner"}},
	{"name": "form_template_requires_name", "description": "Form templates require a readable name.", "condition": {"operation": "create_form_template", "template_name_present": False}, "effect": {"decision": "deny", "reason": "template_name_required", "required_action": "name_form_template"}},
	{"name": "form_template_requires_schema", "description": "Form templates require schema fields.", "condition": {"operation": "create_form_template", "schema_fields_present": False}, "effect": {"decision": "deny", "reason": "schema_validation_required", "required_action": "define_form_schema"}},
	{"name": "form_template_requires_compliance_framework", "description": "Form templates require compliance framework linkage.", "condition": {"operation": "create_form_template", "compliance_framework_present": False}, "effect": {"decision": "deny", "reason": "compliance_framework_link_required", "required_action": "attach_compliance_framework"}},
	{"name": "regulated_form_requires_dlp", "description": "Regulated forms require DLP policy.", "condition": {"operation": "create_form_template", "regulated_form": True, "dlp_policy_present": False}, "effect": {"decision": "deny", "reason": "regulated_field_dlp_required", "required_action": "attach_dlp_policy"}},
	{"name": "form_publication_requires_approval", "description": "Forms require approval before publication.", "condition": {"operation": "publish_form", "publication_approved": False}, "effect": {"decision": "deny", "reason": "publication_approval_required", "required_action": "approve_form_publication"}},
	{"name": "regulated_form_requires_compliance_review", "description": "Regulated forms require compliance review.", "condition": {"regulated_form": True, "compliance_review_recorded": False}, "effect": {"decision": "require_review", "reason": "compliance_review_required", "required_action": "review_regulated_form"}},
	{"name": "submission_requires_evidence", "description": "Form submissions require audit evidence.", "condition": {"operation": "submit_form", "audit_evidence_present": False}, "effect": {"decision": "deny", "reason": "audit_trail_required", "required_action": "attach_submission_audit_ref"}},
	{"name": "submission_requires_valid_schema", "description": "Form submission data must satisfy the template schema.", "condition": {"operation": "submit_form", "schema_valid": False}, "effect": {"decision": "deny", "reason": "schema_validation_required", "required_action": "correct_submission_payload"}},
	{"name": "envelope_requires_subject", "description": "Signature envelopes require a subject.", "condition": {"operation": "create_envelope", "subject_present": False}, "effect": {"decision": "deny", "reason": "envelope_subject_required", "required_action": "add_envelope_subject"}},
	{"name": "envelope_requires_recipient", "description": "Signature envelopes require at least one recipient.", "condition": {"operation": "create_envelope", "recipient_count_lte": 0}, "effect": {"decision": "deny", "reason": "recipient_required", "required_action": "add_recipient"}},
	{"name": "envelope_requires_document_hash", "description": "Signature envelopes require a document hash before routing.", "condition": {"operation": "create_envelope", "document_hash_present": False}, "effect": {"decision": "deny", "reason": "document_hash_required", "required_action": "attach_document_hash"}},
	{"name": "envelope_requires_expiry", "description": "Signature envelopes require an expiry timestamp.", "condition": {"operation": "create_envelope", "expires_at_present": False}, "effect": {"decision": "deny", "reason": "envelope_expiry_required", "required_action": "set_envelope_expiry"}},
	{"name": "envelope_expiry_must_be_future", "description": "Signature envelope expiry must be in the future at send time.", "condition": {"operation": "create_envelope", "expiry_in_future": False}, "effect": {"decision": "deny", "reason": "envelope_expiry_in_past", "required_action": "choose_future_expiry"}},
	{"name": "recipient_requires_consent", "description": "Recipients must consent before routing.", "condition": {"recipient_consent_recorded": False}, "effect": {"decision": "deny", "reason": "recipient_consent_required", "required_action": "record_recipient_consent"}},
	{"name": "delegated_signing_requires_policy", "description": "Delegated signing requires an approved policy reference.", "condition": {"delegated_signer_present": True, "delegated_policy_attached": False}, "effect": {"decision": "deny", "reason": "delegated_signing_policy_required", "required_action": "attach_delegated_signing_policy"}},
	{"name": "signing_requires_identity_verification", "description": "Signing requires verified signer identity.", "condition": {"operation": "sign_envelope", "identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_signer_identity"}},
	{"name": "signing_requires_intent", "description": "Signing requires explicit signature intent.", "condition": {"operation": "sign_envelope", "signature_intent_present": False}, "effect": {"decision": "deny", "reason": "signature_intent_required", "required_action": "record_signature_intent"}},
	{"name": "signing_requires_active_envelope", "description": "Only active envelopes may be signed.", "condition": {"operation": "sign_envelope", "envelope_signable": False}, "effect": {"decision": "deny", "reason": "envelope_not_signable", "required_action": "restore_or_reissue_envelope"}},
	{"name": "signing_requires_order", "description": "Signer routing order must be respected.", "condition": {"operation": "sign_envelope", "routing_order_ready": False}, "effect": {"decision": "deny", "reason": "signer_routing_order_required", "required_action": "wait_for_prior_signers"}},
	{"name": "signing_blocks_duplicate_recipient", "description": "A recipient may sign an envelope only once.", "condition": {"operation": "sign_envelope", "recipient_already_signed": True}, "effect": {"decision": "deny", "reason": "recipient_already_signed", "required_action": "review_existing_signature"}},
	{"name": "signing_requires_valid_seal", "description": "The envelope tamper seal must validate before signing.", "condition": {"operation": "sign_envelope", "tamper_seal_valid": False}, "effect": {"decision": "deny", "reason": "tamper_seal_invalid", "required_action": "rebuild_or_reissue_envelope"}},
	{"name": "signing_blocks_expired_envelope", "description": "Expired envelopes may not be signed.", "condition": {"operation": "sign_envelope", "envelope_expired": True}, "effect": {"decision": "deny", "reason": "envelope_expired", "required_action": "reissue_envelope"}},
	{"name": "evidence_requires_encryption", "description": "Signature evidence packages must be encrypted.", "condition": {"evidence_package_created": True, "evidence_encrypted": False}, "effect": {"decision": "deny", "reason": "evidence_encryption_required", "required_action": "encrypt_evidence_package"}},
	{"name": "evidence_requires_completed_envelope", "description": "Evidence packages require a completed envelope.", "condition": {"operation": "create_evidence_package", "envelope_completed": False}, "effect": {"decision": "deny", "reason": "envelope_not_completed", "required_action": "complete_all_signatures"}},
	{"name": "evidence_requires_valid_seal", "description": "Evidence packages require valid tamper-seal verification.", "condition": {"operation": "create_evidence_package", "tamper_seal_valid": False}, "effect": {"decision": "deny", "reason": "tamper_seal_invalid", "required_action": "investigate_envelope_integrity"}},
	{"name": "evidence_requires_retention", "description": "Evidence packages require retention policy.", "condition": {"operation": "create_evidence_package", "retention_policy_present": False}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "envelope_state_change_requires_reason", "description": "Envelope cancellation and rejection require a reason.", "condition": {"operation": "change_envelope_state", "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "esgn_state_change_requires_audit", "description": "ESGN state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "esgn_audit_event_required", "required_action": "record_esgn_audit_event"}},
	{"name": "signing_agent_requires_registration", "description": "AI signing assistants must be registered.", "condition": {"signing_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "signing_agent_registration_required", "required_action": "register_signing_agent"}},
	{"name": "signing_agent_runtime_supported", "description": "AI signing assistants must use a configured runtime.", "condition": {"signing_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "signing_agent_runtime_not_supported", "required_action": "choose_supported_signing_agent_runtime"}},
	{"name": "signing_agent_requires_scope", "description": "AI signing assistants require envelope or form scope.", "condition": {"signing_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "signing_agent_scope_required", "required_action": "set_signing_agent_scope"}},
	{"name": "signing_agent_requires_disclosure", "description": "AI-assisted signature activity requires visible disclosure.", "condition": {"signing_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "signing_agent_disclosure_required", "required_action": "disclose_signing_agent"}},
	{"name": "cross_tenant_esgn_access_denied", "description": "Digital form and e-sign records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_esgn_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_esgn_mutation_requires_bytewax", "description": "Batch form and signature mutations must use Bytewax event streams.", "condition": {"operation": "batch_esgn_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/esgn/dashboard", "component": "ESGNDashboard", "permission": "esgn:view", "nav_group": "Overview"},
	{"name": "forms", "path": "/esgn/forms", "component": "FormLibrary", "permission": "esgn:create_forms", "nav_group": "Forms"},
	{"name": "builder", "path": "/esgn/builder", "component": "FormBuilder", "permission": "esgn:create_forms", "nav_group": "Forms"},
	{"name": "submissions", "path": "/esgn/submissions", "component": "SubmissionQueue", "permission": "esgn:view", "nav_group": "Forms"},
	{"name": "envelopes", "path": "/esgn/envelopes", "component": "EnvelopeConsole", "permission": "esgn:send_envelopes", "nav_group": "Signatures"},
	{"name": "signing", "path": "/esgn/signing", "component": "SigningRoom", "permission": "esgn:sign", "nav_group": "Signatures"},
	{"name": "agents", "path": "/esgn/agents", "component": "SigningAgentPanel", "permission": "esgn:send_envelopes", "nav_group": "Signatures"},
	{"name": "evidence", "path": "/esgn/evidence", "component": "EvidenceVault", "permission": "esgn:view", "nav_group": "Governance"},
	{"name": "audit", "path": "/esgn/audit", "component": "ESGNAuditTrail", "permission": "esgn:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/esgn/analytics", "component": "ESGNAnalytics", "permission": "esgn:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/esgn/settings", "component": "ESGNSettings", "permission": "esgn:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "esgn_forms_signing",
	"tokens": {
		"color.primary": "#2C5282",
		"color.accent": "#B7791F",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"form_builder": {"icon": "file-pen", "status_indicator": "template-pill", "risk_style": "schema-band"},
		"submission_queue": {"visual": "submission-table", "status_style": "validation-chip"},
		"envelope_console": {"visual": "recipient-routing", "highlight": "signature-chip"},
		"signing_room": {"visual": "signature-ceremony", "status_style": "identity-chip"},
		"agent_panel": {"visual": "assistant-roster", "status_style": "scope-chip"},
		"evidence_vault": {"visual": "sealed-record-list", "status_style": "audit-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "seal-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.esgn.lifecycle",
	"state": ["form_templates", "form_submissions", "signature_envelopes", "signing_ceremonies", "evidence_packages"],
	"events": [
		"template_created",
		"template_published",
		"form_submitted",
		"envelope_sent",
		"envelope_signed",
		"envelope_cancelled",
		"envelope_rejected",
		"evidence_package_created",
		"signing_agent_registered",
	],
	"batch_mutation_guardrail": "batch_esgn_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ESGN capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "esgn",
		"display_name": "Digital Forms and eSign",
		"provides": ["digital_forms", "signature_envelopes", "signing_ceremonies", "evidence_packages", "signing_agent_assist", "form_workflows"],
		"requires": ["auth", "encr", "audl", "comp"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/esgn/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default ESGN governance rules."""
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
