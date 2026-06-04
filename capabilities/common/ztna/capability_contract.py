"""Executable capability contract for APG Zero Trust Network Access."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_ZTNA_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_ZTNA_AGENT_ROLES: list[str] = [
	"policy_reviewer",
	"identity_context_reviewer",
	"device_posture_reviewer",
	"resource_access_reviewer",
	"session_risk_reviewer",
	"segmentation_reviewer",
	"access_review_reviewer",
	"lifecycle_batch_reviewer",
	"zero_trust_steward",
]

PRIVILEGED_ZTNA_AGENT_ROLES: list[str] = [
	"resource_access_reviewer",
	"session_risk_reviewer",
	"segmentation_reviewer",
	"access_review_reviewer",
	"lifecycle_batch_reviewer",
	"zero_trust_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"identities": {
		"verified_identity_required": True,
		"mfa_required_for_privileged": True,
		"federated_identity_allowed": True,
		"continuous_identity_checks": True,
		"suspended_identity_denied": True,
	},
	"devices": {
		"posture_required": True,
		"minimum_device_trust": 0.7,
		"managed_device_preferred": True,
		"attestation_required_for_sensitive_resources": True,
		"quarantine_untrusted_devices": True,
	},
	"resources": {
		"resource_policy_required": True,
		"least_privilege_default": True,
		"session_recording_for_privileged": True,
		"microsegmentation_enabled": True,
		"sensitive_resource_attestation_required": True,
	},
	"access": {
		"deny_by_default": True,
		"risk_threshold": 0.8,
		"high_risk_review_required": True,
		"just_in_time_approval_supported": True,
		"privileged_approval_required": True,
	},
	"sessions": {
		"continuous_verification_required": True,
		"reauth_on_risk_change": True,
		"revocation_supported": True,
		"max_session_hours": 12,
		"close_reason_required": True,
	},
	"segmentation": {
		"network_segment_required": True,
		"default_segment": "default",
		"microsegmentation_required": True,
	},
	"reviews": {
		"independent_reviewer_required": True,
		"review_notes_required": True,
		"duplicate_review_blocked": True,
	},
	"security": {
		"tenant_isolation_required": True,
		"deny_by_default": True,
		"audit_access_decisions": True,
		"policy_mutation_audit_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_access_decisions": True,
		"risk_threshold": 0.8,
		"deny_by_default": True,
	},
	"observability": {
		"audit_required": True,
		"metrics_required": True,
		"trace_required": True,
		"event_stream": "bytewax",
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_ZTNA_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ZTNA_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_ZTNA_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_zero_trust_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "ztna.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"identity_batch",
			"device_posture_batch",
			"resource_batch",
			"access_request_batch",
			"session_batch",
			"review_batch",
			"policy_batch",
			"ztna_agent_batch",
		],
		"topics": [
			"ztna.identities",
			"ztna.devices",
			"ztna.resources",
			"ztna.access",
			"ztna.sessions",
			"ztna.reviews",
			"ztna.policies",
			"ztna.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "service.ZtnaService",
		"helper_runtime": "zero_trust_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"production_runtime": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"authentication": "auth",
		"security_framework": "secu",
		"mfa_provider": "mfau",
		"monitoring": "moni",
		"audit_sink": "audl",
		"identity_federation": "idfd",
		"anomaly_detection": "anom",
		"message_bus": "mqeb",
		"cache": "cach",
		"agent_adapter": "aicr_provider_neutral_zero_trust_agent_adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_policy_console": True,
		"enable_identity_console": True,
		"enable_device_posture": True,
		"enable_resource_map": True,
		"enable_access_console": True,
		"enable_session_monitor": True,
		"enable_risk_console": True,
		"enable_review_queue": True,
		"enable_zero_trust_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "ztna_zero_trust_ops", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"identities",
		"devices",
		"resources",
		"access",
		"sessions",
		"segmentation",
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
		"identities",
		"devices",
		"resources",
		"access",
		"sessions",
		"segmentation",
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
	{"name": "tenant_context_required", "description": "All zero-trust decisions require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "identity_subject_required", "description": "Identity registration requires a subject.", "condition": {"operation": "register_identity", "subject_present": False}, "effect": {"decision": "deny", "reason": "identity_subject_required", "required_action": "attach_subject"}},
	{"name": "identity_display_name_required", "description": "Identity registration requires a display name.", "condition": {"operation": "register_identity", "display_name_present": False}, "effect": {"decision": "deny", "reason": "identity_display_name_required", "required_action": "record_display_name"}},
	{"name": "identity_must_be_verified", "description": "Access requires verified identity.", "condition": {"identity_verified": False}, "effect": {"decision": "deny", "reason": "identity_verification_required", "required_action": "verify_identity"}},
	{"name": "suspended_identity_denied", "description": "Suspended identities cannot access resources.", "condition": {"identity_status": "suspended"}, "effect": {"decision": "deny", "reason": "identity_suspended", "required_action": "restore_identity"}},
	{"name": "federated_identity_requires_provider", "description": "Federated identities require provider evidence.", "condition": {"operation": "register_identity", "federated_identity": True, "federated_provider_present": False}, "effect": {"decision": "deny", "reason": "federated_provider_required", "required_action": "attach_federated_provider"}},
	{"name": "device_requires_identity", "description": "Device registration requires a tenant-local identity.", "condition": {"operation": "register_device", "identity_present": False}, "effect": {"decision": "deny", "reason": "device_identity_required", "required_action": "select_identity"}},
	{"name": "device_posture_required", "description": "Access requires current device posture.", "condition": {"device_posture_present": False}, "effect": {"decision": "deny", "reason": "device_posture_required", "required_action": "collect_device_posture"}},
	{"name": "device_trust_score_requires_threshold", "description": "Device trust score must meet tenant threshold.", "condition": {"device_trust_score_lt": 0.7}, "effect": {"decision": "deny", "reason": "device_trust_too_low", "required_action": "quarantine_or_repair_device"}},
	{"name": "device_compliance_required", "description": "Non-compliant devices cannot access resources.", "condition": {"device_compliant": False}, "effect": {"decision": "deny", "reason": "device_compliance_required", "required_action": "remediate_device"}},
	{"name": "sensitive_resource_requires_attested_device", "description": "Sensitive resources require attested devices.", "condition": {"sensitive_resource": True, "device_attested": False}, "effect": {"decision": "deny", "reason": "device_attestation_required", "required_action": "attest_device"}},
	{"name": "managed_device_preferred_for_privileged", "description": "Privileged access from unmanaged devices requires review.", "condition": {"access_level": "privileged", "managed_device": False, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "managed_device_review_required", "required_action": "review_unmanaged_privileged_access"}},
	{"name": "resource_name_required", "description": "Resource registration requires a name.", "condition": {"operation": "register_resource", "resource_name_present": False}, "effect": {"decision": "deny", "reason": "resource_name_required", "required_action": "name_resource"}},
	{"name": "resource_policy_required", "description": "Resource access requires a matching policy.", "condition": {"resource_policy_attached": False}, "effect": {"decision": "deny", "reason": "resource_policy_required", "required_action": "attach_resource_policy"}},
	{"name": "resource_segment_required", "description": "Resources require a network segment.", "condition": {"operation": "register_resource", "network_segment_present": False}, "effect": {"decision": "deny", "reason": "network_segment_required", "required_action": "assign_network_segment"}},
	{"name": "microsegmentation_required", "description": "Sensitive resources require microsegmentation evidence.", "condition": {"sensitive_resource": True, "microsegmentation_present": False}, "effect": {"decision": "require_review", "reason": "microsegmentation_required", "required_action": "attach_microsegmentation_policy"}},
	{"name": "privileged_access_requires_mfa", "description": "Privileged access requires MFA.", "condition": {"access_level": "privileged", "mfa_completed": False}, "effect": {"decision": "deny", "reason": "privileged_mfa_required", "required_action": "complete_mfa"}},
	{"name": "privileged_access_requires_approval", "description": "Privileged access requires review or approval.", "condition": {"access_level": "privileged", "access_review_recorded": False, "just_in_time_approval_present": False}, "effect": {"decision": "require_review", "reason": "privileged_access_review_required", "required_action": "approve_privileged_access"}},
	{"name": "least_privilege_scope_required", "description": "Access requests require least-privilege scope.", "condition": {"operation": "request_access", "least_privilege_scope_present": False}, "effect": {"decision": "require_review", "reason": "least_privilege_scope_required", "required_action": "narrow_access_scope"}},
	{"name": "high_risk_access_requires_review", "description": "High-risk access decisions require review.", "condition": {"access_risk_score_gt": 0.8, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "high_risk_access_review_required", "required_action": "review_access_request"}},
	{"name": "deny_by_default_requires_explicit_approval", "description": "Deny-by-default posture requires explicit access decision evidence.", "condition": {"operation": "request_access", "explicit_access_decision_present": False}, "effect": {"decision": "require_review", "reason": "explicit_access_decision_required", "required_action": "record_access_decision"}},
	{"name": "session_requires_approved_request", "description": "Sessions require approved access requests.", "condition": {"operation": "start_session", "access_request_approved": False}, "effect": {"decision": "deny", "reason": "access_request_not_approved", "required_action": "approve_access_request"}},
	{"name": "session_requires_continuous_verification", "description": "Active sessions require continuous verification evidence.", "condition": {"operation": "reevaluate_session", "continuous_verification_present": False}, "effect": {"decision": "deny", "reason": "continuous_verification_required", "required_action": "collect_session_context"}},
	{"name": "session_reauth_required_on_high_risk", "description": "High-risk sessions require reauthentication.", "condition": {"operation": "reevaluate_session", "access_risk_score_gt": 0.8, "access_review_recorded": False}, "effect": {"decision": "require_review", "reason": "session_reauth_required", "required_action": "reauthenticate_session"}},
	{"name": "session_close_requires_actor", "description": "Session closure requires an actor.", "condition": {"operation": "close_session", "actor_present": False}, "effect": {"decision": "deny", "reason": "session_close_actor_required", "required_action": "record_session_actor"}},
	{"name": "resource_policy_attachment_requires_policy", "description": "Policy attachment requires a policy ID.", "condition": {"operation": "attach_resource_policy", "policy_present": False}, "effect": {"decision": "deny", "reason": "resource_policy_id_required", "required_action": "select_policy"}},
	{"name": "review_requires_independent_reviewer", "description": "Access reviews require an independent reviewer.", "condition": {"operation": "approve_access_request", "reviewer_same_as_requester": True}, "effect": {"decision": "deny", "reason": "independent_access_review_required", "required_action": "route_to_independent_reviewer"}},
	{"name": "review_decision_requires_notes", "description": "Access reviews require notes.", "condition": {"operation": "approve_access_request", "notes_present": False}, "effect": {"decision": "require_review", "reason": "access_review_notes_required", "required_action": "record_review_notes"}},
	{"name": "duplicate_access_review_blocked", "description": "Duplicate pending access reviews are blocked.", "condition": {"operation": "request_access", "duplicate_pending_review": True}, "effect": {"decision": "deny", "reason": "duplicate_access_review", "required_action": "complete_existing_review"}},
	{"name": "audit_required_for_access_decision", "description": "Zero-trust access decisions require audit evidence.", "condition": {"access_decision_recorded": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "access_audit_event_required", "required_action": "record_access_audit"}},
	{"name": "batch_ztna_mutation_requires_bytewax", "description": "Batch zero-trust mutations must use Bytewax event streams.", "condition": {"operation": "batch_ztna_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "cross_tenant_zero_trust_access_denied", "description": "Zero-trust records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_zero_trust_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "zero_trust_state_change_requires_audit", "description": "Zero-trust state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "zero_trust_audit_event_required", "required_action": "record_zero_trust_audit_event"}},
	{"name": "ztna_agent_runtime_supported", "description": "Zero-trust agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_zero_trust_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ztna_agent_runtime", "required_action": "choose_supported_ztna_agent_runtime"}},
	{"name": "ztna_agent_role_supported", "description": "Zero-trust agents must use supported access-governance roles.", "condition": {"operation": "register_zero_trust_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ztna_agent_role", "required_action": "choose_supported_ztna_agent_role"}},
	{"name": "ztna_agent_requires_scope", "description": "Zero-trust agents require an explicit identity, device, resource, access, session, policy, segment, review, or lifecycle scope.", "condition": {"operation": "register_zero_trust_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "ztna_agent_scope_required", "required_action": "declare_ztna_agent_scope"}},
	{"name": "ztna_agent_requires_owner", "description": "Zero-trust agents require an accountable owner.", "condition": {"operation": "register_zero_trust_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "ztna_agent_owner_required", "required_action": "assign_ztna_agent_owner"}},
	{"name": "ztna_agent_requires_purpose", "description": "Zero-trust agents require a documented access-governance purpose.", "condition": {"operation": "register_zero_trust_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "ztna_agent_purpose_required", "required_action": "document_ztna_agent_purpose"}},
	{"name": "ztna_agent_requires_contribution_disclosure", "description": "Zero-trust agents must disclose machine-authored identity, device, resource, access, session, policy, segment, review, and lifecycle contributions.", "condition": {"operation": "register_zero_trust_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ztna_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "ztna_agent_privileged_role_requires_human_approval", "description": "Privileged zero-trust agent roles require human approval evidence.", "condition": {"operation": "register_zero_trust_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "ztna_agent_human_approval_required", "required_action": "record_human_ztna_agent_approval"}},
	{"name": "ztna_lifecycle_batch_requires_mutations", "description": "ZTNA lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_ztna_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "ztna_lifecycle_batch_empty", "required_action": "include_ztna_lifecycle_mutations"}},
	{"name": "bytewax_ztna_stream_required", "description": "ZTNA lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_ztna_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_ztna_lifecycle_batch_to_bytewax"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/ztna/dashboard", "component": "ZTNADashboard", "permission": "ztna:view", "nav_group": "Overview"},
	{"name": "policies", "path": "/ztna/policies", "component": "ZeroTrustPolicies", "permission": "ztna:manage_policies", "nav_group": "Policies"},
	{"name": "identities", "path": "/ztna/identities", "component": "IdentityConsole", "permission": "ztna:manage_policies", "nav_group": "Identity"},
	{"name": "devices", "path": "/ztna/devices", "component": "DevicePosture", "permission": "ztna:manage_devices", "nav_group": "Devices"},
	{"name": "resources", "path": "/ztna/resources", "component": "ResourceMap", "permission": "ztna:manage_policies", "nav_group": "Resources"},
	{"name": "access", "path": "/ztna/access", "component": "AccessRequests", "permission": "ztna:approve_access", "nav_group": "Access"},
	{"name": "sessions", "path": "/ztna/sessions", "component": "SessionMonitor", "permission": "ztna:view", "nav_group": "Operations"},
	{"name": "risk", "path": "/ztna/risk", "component": "AccessRiskConsole", "permission": "ztna:view", "nav_group": "Operations"},
	{"name": "reviews", "path": "/ztna/reviews", "component": "AccessReviewQueue", "permission": "ztna:review", "nav_group": "Governance"},
	{"name": "agents", "path": "/ztna/agents", "component": "ZeroTrustAgentRoster", "permission": "ztna:admin", "nav_group": "Governance"},
	{"name": "lifecycle", "path": "/ztna/lifecycle", "component": "ZTNALifecycleBatchMonitor", "permission": "ztna:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/ztna/audit", "component": "ZTNAAuditTrail", "permission": "ztna:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/ztna/settings", "component": "ZTNASettings", "permission": "ztna:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "ztna_zero_trust_ops",
	"tokens": {
		"color.primary": "#1A365D",
		"color.accent": "#38A169",
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
		"access_decision": {"icon": "shield", "status_indicator": "decision-pill", "risk_style": "trust-band"},
		"identity_console": {"visual": "identity-list", "status_style": "verification-chip"},
		"device_posture": {"visual": "posture-checklist", "highlight": "trust-score-chip"},
		"resource_map": {"visual": "segmented-network-map", "status_style": "policy-chip"},
		"session_monitor": {"visual": "active-session-table", "status_style": "reauth-chip"},
		"risk_console": {"visual": "risk-lanes", "status_style": "risk-chip"},
		"review_queue": {"visual": "decision-lane", "status_style": "review-chip"},
		"zero_trust_agent_roster": {"icon": "bot", "status_indicator": "agent-approval-chip", "risk_style": "access-scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "decision-chip"},
	},
}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.ztna.lifecycle",
	"key": "tenant_id",
	"events": [
		"policy_created",
		"policy_updated",
		"policy_activated",
		"policy_deactivated",
		"access_granted",
		"access_denied",
		"session_established",
		"session_terminated",
		"trust_score_evaluated",
		"device_posture_checked",
		"micro_segment_created",
		"agent_registered",
	],
	"guardrails": [
		"ztna_batch_requires_bytewax",
		"ztna_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable ZTNA capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "ztna",
		"display_name": "Zero Trust Network Access",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"agents": agent_manifest(),
		"streaming": deepcopy(STREAMING),
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/ztna/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
	}


def agent_manifest() -> dict[str, Any]:
	"""Return provider-neutral zero-trust agent composition metadata."""
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
	"""Return Bytewax lifecycle stream metadata for ZTNA composition."""
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
	"""Evaluate default ZTNA governance rules."""
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
