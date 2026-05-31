"""Executable capability contract for APG Facial Recognition."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_FREC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_FREC_AGENT_ROLES = [
	"consent_reviewer",
	"enrollment_reviewer",
	"liveness_reviewer",
	"verification_reviewer",
	"watchlist_reviewer",
	"identification_reviewer",
	"emotion_governance_reviewer",
	"privacy_reviewer",
	"lifecycle_batch_reviewer",
	"facial_recognition_steward",
]
PRIVILEGED_FREC_AGENT_ROLES = [
	"liveness_reviewer",
	"verification_reviewer",
	"watchlist_reviewer",
	"identification_reviewer",
	"emotion_governance_reviewer",
	"privacy_reviewer",
	"lifecycle_batch_reviewer",
	"facial_recognition_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"consent": {
		"explicit_consent_required": True,
		"purpose_required": True,
		"revocation_supported": True,
	},
	"recognition": {
		"enabled_modes": ["verification", "identification", "watchlist_matching"],
		"minimum_face_quality": 0.72,
		"verification_threshold": 0.88,
		"identification_threshold": 0.92,
		"low_confidence_review_threshold": 0.8,
	},
	"enrollment": {
		"active_consent_required": True,
		"template_hash_required": True,
		"template_encryption_required": True,
		"retention_policy_required": True,
	},
	"templates": {
		"encrypted_storage_required": True,
		"raw_image_retention": "disabled",
		"template_rotation_days": 365,
		"retirement_supported": True,
	},
	"liveness": {
		"required_for_authentication": True,
		"minimum_liveness_score": 0.84,
		"anti_spoofing_enabled": True,
		"deepfake_detection_enabled": True,
	},
	"verification": {
		"active_template_required": True,
		"subject_template_match_required": True,
		"liveness_required": True,
		"review_low_confidence": True,
	},
	"identification": {
		"watchlist_policy_required": True,
		"review_watchlist_hits": True,
		"review_low_confidence_matches": True,
	},
	"watchlists": {
		"policy_required": True,
		"owner_required": True,
		"reason_required": True,
		"audit_membership_changes": True,
	},
	"emotion": {
		"emotion_analysis_enabled": False,
		"explicit_purpose_required": True,
		"aggregate_only_by_default": True,
		"raw_emotion_retention": "disabled",
	},
	"privacy": {
		"explicit_consent_required": True,
		"watchlist_policy_required": True,
		"audit_identification": True,
		"template_encryption_required": True,
		"privacy_review_for_sensitive_watchlists": True,
	},
	"reviews": {
		"independent_reviewer_required": True,
		"review_notes_required": True,
		"duplicate_pending_review_blocked": True,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_FREC_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_FREC_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_FREC_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_facial_recognition_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "frec.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"face_consent_batch",
			"face_template_batch",
			"liveness_batch",
			"verification_batch",
			"watchlist_batch",
			"identification_batch",
			"emotion_batch",
			"face_review_batch",
			"facial_recognition_agent_batch",
		],
		"topics": [
			"frec.consents",
			"frec.templates",
			"frec.liveness",
			"frec.verifications",
			"frec.watchlists",
			"frec.identifications",
			"frec.emotion",
			"frec.reviews",
			"frec.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"security": {
		"tenant_isolation_required": True,
		"raw_face_image_retention_allowed": False,
		"audit_state_changes": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_enrollment": True,
		"audit_verification": True,
		"audit_identification": True,
		"audit_watchlists": True,
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "face_runtime.FrecService",
		"helper_runtime": "face_runtime.py",
		"api_helpers": "api_helpers.py",
		"view_models": "view_models.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"biometric_processing": "biop",
		"computer_vision": "cvsn",
		"ai_core": "aicr",
		"encryption": "encr",
		"audit_sink": "audl",
		"mfa_provider": "mfau",
		"cache": "cach",
		"metrics_sink": "moni",
		"agent_adapter": "aicr_provider_neutral_facial_recognition_agent_adapter",
	},
	"ui": {
		"enable_identity_dashboard": True,
		"enable_subject_registry": True,
		"enable_consent_center": True,
		"enable_enrollment_console": True,
		"enable_template_gallery": True,
		"enable_verification_workbench": True,
		"enable_identification_workbench": True,
		"enable_liveness_console": True,
		"enable_watchlist_manager": True,
		"enable_review_queue": True,
		"enable_emotion_governance": True,
		"enable_facial_recognition_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "frec_identity_vision", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"consent",
		"recognition",
		"enrollment",
		"templates",
		"liveness",
		"verification",
		"identification",
		"watchlists",
		"emotion",
		"privacy",
		"reviews",
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
		"consent",
		"recognition",
		"enrollment",
		"templates",
		"liveness",
		"verification",
		"identification",
		"watchlists",
		"emotion",
		"privacy",
		"reviews",
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
	{"name": "tenant_context_required", "description": "All facial recognition operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "face_consent_requires_subject", "description": "Face consent requires a subject.", "condition": {"operation": "record_face_consent", "subject_present": False}, "effect": {"decision": "deny", "reason": "face_subject_required", "required_action": "select_subject"}},
	{"name": "face_consent_requires_purpose", "description": "Face consent requires a purpose.", "condition": {"operation": "record_face_consent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "face_consent_purpose_required", "required_action": "record_consent_purpose"}},
	{"name": "face_consent_requires_evidence", "description": "Face consent requires evidence.", "condition": {"operation": "record_face_consent", "evidence_present": False}, "effect": {"decision": "deny", "reason": "face_consent_evidence_required", "required_action": "attach_consent_evidence"}},
	{"name": "face_enrollment_requires_consent", "description": "Face enrollment requires explicit consent.", "condition": {"operation": "enroll_face", "consent_recorded": False}, "effect": {"decision": "deny", "reason": "face_consent_required", "required_action": "record_face_consent"}},
	{"name": "face_enrollment_requires_active_consent", "description": "Face enrollment requires active consent.", "condition": {"operation": "enroll_face", "active_consent_present": False}, "effect": {"decision": "deny", "reason": "active_face_consent_required", "required_action": "record_or_restore_face_consent"}},
	{"name": "face_template_requires_hash", "description": "Face enrollment requires a template hash.", "condition": {"operation": "enroll_face", "template_hash_present": False}, "effect": {"decision": "deny", "reason": "face_template_hash_required", "required_action": "attach_face_template_hash"}},
	{"name": "face_template_requires_encryption", "description": "Face templates require encryption evidence.", "condition": {"operation": "enroll_face", "template_encrypted": False}, "effect": {"decision": "deny", "reason": "face_template_encryption_required", "required_action": "encrypt_face_template"}},
	{"name": "face_quality_requires_threshold", "description": "Face quality must meet tenant threshold.", "condition": {"operation": "enroll_face", "face_quality_lt": 0.72}, "effect": {"decision": "deny", "reason": "face_quality_too_low", "required_action": "recapture_face"}},
	{"name": "low_face_quality_requires_recapture", "description": "Low-quality face captures require recapture or review.", "condition": {"face_quality_lt": 0.72, "recapture_completed": False}, "effect": {"decision": "require_review", "reason": "low_face_quality", "required_action": "recapture_or_review"}},
	{"name": "raw_face_image_retention_denied", "description": "Raw face images may not be retained in the package runtime.", "condition": {"raw_face_image_retention_requested": True}, "effect": {"decision": "deny", "reason": "raw_face_image_retention_denied", "required_action": "store_template_metadata_only"}},
	{"name": "verification_requires_active_template", "description": "Face verification requires an active template.", "condition": {"operation": "verify_face", "active_template_present": False}, "effect": {"decision": "deny", "reason": "active_face_template_required", "required_action": "enroll_active_face_template"}},
	{"name": "verification_requires_subject_template_match", "description": "Face verification subject must match template subject.", "condition": {"operation": "verify_face", "subject_matches_template": False}, "effect": {"decision": "deny", "reason": "face_template_subject_mismatch", "required_action": "select_subject_template"}},
	{"name": "verification_requires_liveness", "description": "Face verification requires liveness evidence.", "condition": {"operation": "verify_face", "liveness_present": False}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "authentication_requires_liveness", "description": "Face authentication requires liveness evidence.", "condition": {"operation": "authenticate_face", "liveness_passed": False}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "liveness_score_requires_threshold", "description": "Liveness score must meet tenant threshold.", "condition": {"operation": "authenticate_face", "liveness_score_lt": 0.84}, "effect": {"decision": "deny", "reason": "liveness_required", "required_action": "complete_liveness_check"}},
	{"name": "spoof_signal_blocks_face_authentication", "description": "Spoof signals block facial authentication.", "condition": {"operation": "authenticate_face", "spoof_detected": True}, "effect": {"decision": "deny", "reason": "face_spoof_detected", "required_action": "escalate_security_review"}},
	{"name": "deepfake_signal_blocks_face_authentication", "description": "Deepfake signals block facial authentication.", "condition": {"operation": "authenticate_face", "deepfake_detected": True}, "effect": {"decision": "deny", "reason": "face_deepfake_detected", "required_action": "escalate_security_review"}},
	{"name": "verification_confidence_requires_threshold", "description": "Face verification confidence must meet tenant threshold.", "condition": {"operation": "verify_face", "match_confidence_lt": 0.88, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_face_match_confidence", "required_action": "review_face_match"}},
	{"name": "identification_requires_watchlist_policy", "description": "Identification requires an active watchlist policy.", "condition": {"operation": "identify_face", "watchlist_policy_attached": False}, "effect": {"decision": "deny", "reason": "watchlist_policy_required", "required_action": "attach_watchlist_policy"}},
	{"name": "watchlist_requires_owner", "description": "Watchlists require an owner.", "condition": {"operation": "create_watchlist", "owner_present": False}, "effect": {"decision": "deny", "reason": "watchlist_owner_required", "required_action": "assign_watchlist_owner"}},
	{"name": "watchlist_requires_reason", "description": "Watchlists require a reason.", "condition": {"operation": "create_watchlist", "reason_present": False}, "effect": {"decision": "deny", "reason": "watchlist_reason_required", "required_action": "record_watchlist_reason"}},
	{"name": "watchlist_subject_requires_active_template", "description": "Watchlist subjects require active face templates.", "condition": {"operation": "add_watchlist_subject", "active_template_present": False}, "effect": {"decision": "deny", "reason": "active_face_template_required", "required_action": "enroll_active_face_template"}},
	{"name": "watchlist_hit_requires_review", "description": "Watchlist hits require review evidence.", "condition": {"operation": "identify_face", "watchlist_hit": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "watchlist_hit_review_required", "required_action": "review_watchlist_hit"}},
	{"name": "identification_confidence_requires_threshold", "description": "Face identification confidence must meet tenant threshold.", "condition": {"operation": "identify_face", "identification_confidence_lt": 0.92, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "low_identification_confidence", "required_action": "review_identification_match"}},
	{"name": "emotion_analysis_requires_explicit_purpose", "description": "Emotion analysis requires an explicit approved purpose.", "condition": {"emotion_analysis_requested": True, "approved_purpose_recorded": False}, "effect": {"decision": "deny", "reason": "emotion_purpose_required", "required_action": "record_approved_purpose"}},
	{"name": "emotion_analysis_requires_aggregate_mode", "description": "Emotion analysis defaults to aggregate-only outputs.", "condition": {"operation": "analyze_emotion", "aggregate_only": False, "individual_emotion_approval_recorded": False}, "effect": {"decision": "deny", "reason": "individual_emotion_approval_required", "required_action": "record_individual_emotion_approval"}},
	{"name": "review_requires_independent_reviewer", "description": "Face recognition reviews require an independent reviewer.", "condition": {"operation": "decide_review", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_face_review_required", "required_action": "route_to_independent_reviewer"}},
	{"name": "review_decision_requires_notes", "description": "Face recognition review decisions require notes.", "condition": {"operation": "decide_review", "notes_present": False}, "effect": {"decision": "deny", "reason": "face_review_notes_required", "required_action": "record_review_notes"}},
	{"name": "duplicate_pending_review_blocked", "description": "Duplicate pending face recognition reviews are blocked.", "condition": {"operation": "request_review", "pending_review_exists": True}, "effect": {"decision": "deny", "reason": "face_review_already_pending", "required_action": "complete_existing_review"}},
	{"name": "template_retirement_requires_reason", "description": "Face template retirement requires a reason.", "condition": {"operation": "retire_template", "retirement_reason_present": False}, "effect": {"decision": "deny", "reason": "face_template_retirement_reason_required", "required_action": "record_retirement_reason"}},
	{"name": "consent_revocation_retires_templates", "description": "Face consent revocation requires active templates to be retired.", "condition": {"operation": "revoke_face_consent", "active_templates_retired": False}, "effect": {"decision": "deny", "reason": "face_consent_revocation_requires_template_retirement", "required_action": "retire_templates_for_consent"}},
	{"name": "batch_face_mutation_requires_bytewax", "description": "Batch facial recognition mutations must use Bytewax event streams.", "condition": {"operation": "batch_face_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_face_access_denied", "description": "Facial recognition records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_face_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "face_state_change_requires_audit", "description": "Facial recognition state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "face_audit_event_required", "required_action": "record_face_audit_event"}},
	{"name": "facial_recognition_agent_runtime_supported", "description": "Facial-recognition agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_facial_recognition_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_facial_recognition_agent_runtime", "required_action": "choose_supported_frec_agent_runtime"}},
	{"name": "facial_recognition_agent_role_supported", "description": "Facial-recognition agents must use supported identity-governance roles.", "condition": {"operation": "register_facial_recognition_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_facial_recognition_agent_role", "required_action": "choose_supported_frec_agent_role"}},
	{"name": "facial_recognition_agent_requires_scope", "description": "Facial-recognition agents require an explicit consent, enrollment, liveness, verification, watchlist, identification, emotion, privacy, or lifecycle scope.", "condition": {"operation": "register_facial_recognition_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "facial_recognition_agent_scope_required", "required_action": "declare_frec_agent_scope"}},
	{"name": "facial_recognition_agent_requires_owner", "description": "Facial-recognition agents require an accountable owner.", "condition": {"operation": "register_facial_recognition_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "facial_recognition_agent_owner_required", "required_action": "assign_frec_agent_owner"}},
	{"name": "facial_recognition_agent_requires_purpose", "description": "Facial-recognition agents require a documented purpose.", "condition": {"operation": "register_facial_recognition_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "facial_recognition_agent_purpose_required", "required_action": "document_frec_agent_purpose"}},
	{"name": "facial_recognition_agent_requires_contribution_disclosure", "description": "Facial-recognition agents must disclose machine-authored enrollment, liveness, match, watchlist, emotion, privacy, and lifecycle-review contributions.", "condition": {"operation": "register_facial_recognition_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "facial_recognition_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "facial_recognition_agent_privileged_role_requires_human_approval", "description": "Privileged facial-recognition agent roles require human approval evidence.", "condition": {"operation": "register_facial_recognition_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "facial_recognition_agent_human_approval_required", "required_action": "record_human_frec_agent_approval"}},
	{"name": "frec_lifecycle_batch_requires_mutations", "description": "FREC lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_frec_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "frec_lifecycle_batch_empty", "required_action": "include_frec_lifecycle_mutations"}},
	{"name": "bytewax_frec_stream_required", "description": "FREC lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_frec_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_frec_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/frec/dashboard", "component": "FRECDashboard", "permission": "frec:view", "nav_group": "Overview"},
	{"name": "subjects", "path": "/frec/subjects", "component": "FaceSubjectRegistry", "permission": "frec:view", "nav_group": "Identity"},
	{"name": "consents", "path": "/frec/consents", "component": "FaceConsentCenter", "permission": "frec:enroll", "nav_group": "Identity"},
	{"name": "enrollment", "path": "/frec/enrollment", "component": "FaceEnrollment", "permission": "frec:enroll", "nav_group": "Identity"},
	{"name": "templates", "path": "/frec/templates", "component": "FaceTemplateGallery", "permission": "frec:enroll", "nav_group": "Identity"},
	{"name": "verification", "path": "/frec/verification", "component": "FaceVerification", "permission": "frec:verify", "nav_group": "Identity"},
	{"name": "identification", "path": "/frec/identification", "component": "FaceIdentification", "permission": "frec:identify", "nav_group": "Identity"},
	{"name": "liveness", "path": "/frec/liveness", "component": "FaceLiveness", "permission": "frec:verify", "nav_group": "Security"},
	{"name": "watchlists", "path": "/frec/watchlists", "component": "WatchlistManager", "permission": "frec:manage_watchlists", "nav_group": "Governance"},
	{"name": "reviews", "path": "/frec/reviews", "component": "FaceReviewQueue", "permission": "frec:review", "nav_group": "Governance"},
	{"name": "emotion", "path": "/frec/emotion", "component": "EmotionGovernance", "permission": "frec:admin", "nav_group": "Governance"},
	{"name": "agents", "path": "/frec/agents", "component": "FacialRecognitionAgentRoster", "permission": "frec:admin", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/frec/lifecycle", "component": "FRECLifecycleBatchMonitor", "permission": "frec:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/frec/audit", "component": "FRECAuditTrail", "permission": "frec:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/frec/settings", "component": "FRECSettings", "permission": "frec:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "frec_identity_vision",
	"tokens": {
		"color.primary": "#234E70",
		"color.accent": "#C05621",
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
		"face_quality_panel": {"icon": "scan-face", "status_indicator": "quality-pill", "risk_style": "capture-band"},
		"consent_scope": {"visual": "scope-ledger", "status_style": "consent-chip"},
		"template_gallery": {"visual": "encrypted-face-grid", "status_style": "template-chip"},
		"match_gallery": {"visual": "ranked-face-grid", "highlight": "confidence-chip"},
		"liveness_trace": {"visual": "challenge-timeline", "status_style": "spoof-chip"},
		"watchlist_table": {"visual": "identity-list", "status_style": "policy-chip"},
		"review_queue": {"visual": "decision-lane", "status_style": "review-chip"},
		"emotion_governance": {"visual": "purpose-ledger", "status_style": "aggregate-chip"},
		"facial_recognition_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "face-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "identity-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class FREC agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_FREC_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_FREC_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_FREC_AGENT_ROLES),
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
		"adapter_contract": "aicr_provider_neutral_facial_recognition_agent_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return the FREC Bytewax lifecycle stream contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "frec.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"face_consent_batch",
			"face_template_batch",
			"liveness_batch",
			"verification_batch",
			"watchlist_batch",
			"identification_batch",
			"emotion_batch",
			"face_review_batch",
			"facial_recognition_agent_batch",
		],
		"topics": [
			"frec.consents",
			"frec.templates",
			"frec.liveness",
			"frec.verifications",
			"frec.watchlists",
			"frec.identifications",
			"frec.emotion",
			"frec.reviews",
			"frec.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable FREC capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "frec",
		"display_name": "Facial Recognition",
		"provides": ["facial_recognition", "face_identification", "facial_recognition_agent_composition"],
		"requires": ["biop", "cvsn", "aicr", "encr", "audl", "conf", "mfau"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "view_models.py",
			"api_prefix": "/frec/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default FREC governance rules."""
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
