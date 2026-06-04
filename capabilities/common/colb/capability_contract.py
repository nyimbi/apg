"""Executable capability contract for APG Collaboration Tools."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_COLB_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_COLB_AGENT_ROLES: list[str] = [
	"workspace_reviewer",
	"session_reviewer",
	"artifact_reviewer",
	"annotation_reviewer",
	"decision_reviewer",
	"presence_reviewer",
	"protocol_reviewer",
	"guest_access_reviewer",
	"lifecycle_batch_reviewer",
	"collaboration_steward",
]

PRIVILEGED_COLB_AGENT_ROLES: list[str] = [
	"artifact_reviewer",
	"decision_reviewer",
	"protocol_reviewer",
	"guest_access_reviewer",
	"lifecycle_batch_reviewer",
	"collaboration_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"workspaces": {
		"workspace_owner_required": True,
		"guest_access_supported": True,
		"max_participants_per_workspace": 1000,
		"artifact_policy_required": True,
		"membership_review_threshold": 1000,
		"external_collaboration_policy_required": True,
	},
	"sessions": {
		"realtime_sync_enabled": True,
		"session_recording_supported": True,
		"presence_required": True,
		"conflict_resolution": "operational_transform",
		"secure_transport_required": True,
		"session_owner_required": True,
	},
	"artifacts": {
		"shared_artifacts_enabled": True,
		"artifact_policy_required": True,
		"version_history_required": True,
		"external_share_dlp_required": True,
		"lock_timeout_minutes": 30,
	},
	"annotations": {
		"threaded_annotations_enabled": True,
		"decision_records_enabled": True,
		"annotation_author_required": True,
		"resolution_evidence_required": True,
	},
	"presence": {
		"presence_required": True,
		"cursor_presence_enabled": True,
		"typing_presence_enabled": True,
		"presence_ttl_seconds": 90,
	},
	"protocols": {
		"enabled": ["websocket", "webrtc", "mqtt", "grpc"],
		"secure_transport_required": True,
		"protocol_health_required": True,
		"fallback_protocol_enabled": True,
		"event_stream": "bytewax",
	},
	"ai_agents": {
		"agent_collaboration_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_COLB_AGENT_RUNTIMES,
	},
	"security": {
		"tenant_isolation_required": True,
		"authenticated_actor_required": True,
		"audit_collaboration_events": True,
		"secret_redaction_required": True,
		"external_share_dlp_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_collaboration_events": True,
		"external_collaboration_policy_required": True,
		"retention_policy_required": True,
		"decision_evidence_required": True,
	},
	"retention": {
		"default_policy": "retain-180-days",
		"workspace_retention_required": True,
		"session_recording_retention_required": True,
		"export_requires_approval": True,
	},
	"observability": {
		"workspace_metrics_required": True,
		"session_metrics_required": True,
		"protocol_health_required": True,
		"audit_required": True,
		"event_stream": "bytewax",
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_COLB_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_COLB_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_COLB_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_collaboration_agent_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "colb.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"workspace_batch",
			"session_batch",
			"artifact_batch",
			"annotation_batch",
			"decision_batch",
			"presence_batch",
			"protocol_batch",
			"guest_access_batch",
			"collaboration_agent_batch",
		],
		"topics": [
			"colb.workspaces",
			"colb.sessions",
			"colb.artifacts",
			"colb.annotations",
			"colb.decisions",
			"colb.presence",
			"colb.protocols",
			"colb.guests",
			"colb.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "collaboration_runtime.CollaborationRuntime",
		"helper_runtime": "collaboration_runtime.py",
		"api_helpers": "package_api.py",
		"view_models": "view_models.py",
		"production_app": "production_app.py",
		"production_service": "service.py",
		"production_api": "api.py",
		"production_views": "views.py",
		"event_stream": "bytewax",
		"message_bus": "mqeb",
		"chat": "chat",
		"notification": "ntfy",
		"authentication": "auth",
		"multi_tenancy": "mten",
		"audit_sink": "audl",
		"workflow": "wflo",
		"video": "vidc",
		"nlp": "nlpc",
		"security": "secu",
		"cache": "cach",
		"ai_orchestration": "aicr",
		"agent_adapter": "aicr_provider_neutral_collaboration_agent_adapter",
	},
	"ui": {
		"enable_workspace_dashboard": True,
		"enable_workspace_manager": True,
		"enable_session_console": True,
		"enable_presence_panel": True,
		"enable_artifact_board": True,
		"enable_annotation_panel": True,
		"enable_decision_log": True,
		"enable_agent_panel": True,
		"enable_collaboration_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_protocol_monitor": True,
		"enable_analytics": True,
		"enable_audit": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "colb_collaboration_workspace", "allow_tenant_overrides": True},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"workspaces",
		"sessions",
		"artifacts",
		"annotations",
		"presence",
		"protocols",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"workspaces",
		"sessions",
		"artifacts",
		"annotations",
		"presence",
		"protocols",
		"ai_agents",
		"security",
		"governance",
		"retention",
		"observability",
		"agents",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All collaboration operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "workspace_requires_owner", "description": "Workspaces require an accountable owner.", "condition": {"operation": "create_workspace", "workspace_owner_assigned": False}, "effect": {"decision": "deny", "reason": "workspace_owner_required", "required_action": "assign_workspace_owner"}},
	{"name": "workspace_requires_name", "description": "Workspaces require a readable name.", "condition": {"operation": "create_workspace", "workspace_name_present": False}, "effect": {"decision": "deny", "reason": "workspace_name_required", "required_action": "name_workspace"}},
	{"name": "workspace_requires_participant", "description": "Workspaces require at least one participant or owner.", "condition": {"operation": "create_workspace", "participant_present": False}, "effect": {"decision": "deny", "reason": "workspace_participant_required", "required_action": "add_workspace_participant"}},
	{"name": "workspace_requires_retention", "description": "Workspaces require retention policy.", "condition": {"operation": "create_workspace", "retention_policy_attached": False}, "effect": {"decision": "deny", "reason": "retention_policy_required", "required_action": "attach_retention_policy"}},
	{"name": "external_collaboration_requires_policy", "description": "External collaboration requires a tenant policy.", "condition": {"external_participant_present": True, "external_policy_attached": False}, "effect": {"decision": "deny", "reason": "external_policy_required", "required_action": "attach_external_collaboration_policy"}},
	{"name": "external_collaboration_requires_expiry", "description": "External participants require access expiry.", "condition": {"external_participant_present": True, "external_access_expiry_present": False}, "effect": {"decision": "require_review", "reason": "external_access_expiry_required", "required_action": "set_external_access_expiry"}},
	{"name": "large_workspace_requires_review", "description": "Large workspaces require membership review.", "condition": {"participant_count_gt": 1000, "membership_review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_workspace_review_required", "required_action": "review_workspace_membership"}},
	{"name": "session_requires_workspace", "description": "Realtime sessions require a tenant-local workspace.", "condition": {"operation": "start_session", "workspace_present": False}, "effect": {"decision": "deny", "reason": "workspace_required", "required_action": "select_workspace"}},
	{"name": "session_requires_owner", "description": "Realtime sessions require an owner.", "condition": {"operation": "start_session", "session_owner_assigned": False}, "effect": {"decision": "deny", "reason": "session_owner_required", "required_action": "assign_session_owner"}},
	{"name": "session_owner_membership_required", "description": "Realtime session owners must belong to the workspace.", "condition": {"operation": "start_session", "session_owner_is_member": False}, "effect": {"decision": "deny", "reason": "session_owner_not_workspace_member", "required_action": "add_session_owner_to_workspace"}},
	{"name": "session_requires_active_workspace", "description": "Realtime sessions require active workspaces.", "condition": {"operation": "start_session", "workspace_active": False}, "effect": {"decision": "deny", "reason": "workspace_not_active", "required_action": "activate_workspace"}},
	{"name": "session_requires_secure_transport", "description": "Realtime sessions require secure transport.", "condition": {"realtime_session": True, "secure_transport": False}, "effect": {"decision": "deny", "reason": "secure_transport_required", "required_action": "enable_secure_transport"}},
	{"name": "session_requires_protocol_health", "description": "Realtime sessions require healthy protocols.", "condition": {"realtime_session": True, "protocol_health": "unhealthy"}, "effect": {"decision": "deny", "reason": "protocol_unhealthy", "required_action": "restore_or_fallback_protocol"}},
	{"name": "session_requires_event_bus", "description": "Realtime sessions require event bus evidence.", "condition": {"realtime_session": True, "event_bus_present": False}, "effect": {"decision": "deny", "reason": "event_bus_required", "required_action": "attach_event_bus"}},
	{"name": "participant_membership_required", "description": "Session participants must belong to the workspace.", "condition": {"operation": "join_session", "participant_is_member": False}, "effect": {"decision": "deny", "reason": "participant_not_workspace_member", "required_action": "add_workspace_member"}},
	{"name": "presence_requires_session", "description": "Presence updates require an active collaboration session.", "condition": {"operation": "update_presence", "session_active": False}, "effect": {"decision": "deny", "reason": "session_not_active", "required_action": "start_or_join_session"}},
	{"name": "presence_requires_participant", "description": "Presence updates require a session participant.", "condition": {"operation": "update_presence", "participant_is_member": False}, "effect": {"decision": "deny", "reason": "presence_participant_required", "required_action": "join_session"}},
	{"name": "artifact_policy_required", "description": "Shared artifacts require an artifact policy.", "condition": {"operation": "share_artifact", "artifact_policy_attached": False}, "effect": {"decision": "deny", "reason": "artifact_policy_required", "required_action": "attach_artifact_policy"}},
	{"name": "artifact_requires_version_history", "description": "Shared artifacts require version history.", "condition": {"operation": "share_artifact", "version_history_enabled": False}, "effect": {"decision": "deny", "reason": "artifact_version_history_required", "required_action": "enable_version_history"}},
	{"name": "external_artifact_requires_dlp", "description": "Externally shared artifacts require DLP review.", "condition": {"external_share_requested": True, "dlp_check_completed": False}, "effect": {"decision": "deny", "reason": "dlp_check_required", "required_action": "run_dlp_check"}},
	{"name": "annotation_requires_artifact", "description": "Annotations require a shared artifact.", "condition": {"operation": "add_annotation", "artifact_present": False}, "effect": {"decision": "deny", "reason": "artifact_required", "required_action": "select_artifact"}},
	{"name": "annotation_requires_author", "description": "Annotations require an author.", "condition": {"operation": "add_annotation", "annotation_author_present": False}, "effect": {"decision": "deny", "reason": "annotation_author_required", "required_action": "identify_annotation_author"}},
	{"name": "annotation_requires_body", "description": "Annotations require content.", "condition": {"operation": "add_annotation", "annotation_body_present": False}, "effect": {"decision": "deny", "reason": "annotation_body_required", "required_action": "enter_annotation_body"}},
	{"name": "decision_requires_annotation", "description": "Decision records require an annotation thread.", "condition": {"operation": "record_decision", "annotation_present": False}, "effect": {"decision": "deny", "reason": "annotation_required", "required_action": "select_annotation"}},
	{"name": "decision_requires_owner", "description": "Decision records require an accountable owner.", "condition": {"operation": "record_decision", "decision_owner_present": False}, "effect": {"decision": "deny", "reason": "decision_owner_required", "required_action": "assign_decision_owner"}},
	{"name": "decision_requires_evidence", "description": "Decision records require evidence.", "condition": {"operation": "record_decision", "decision_evidence_present": False}, "effect": {"decision": "deny", "reason": "decision_evidence_required", "required_action": "attach_decision_evidence"}},
	{"name": "recording_requires_retention", "description": "Session recordings require retention policy.", "condition": {"recording_requested": True, "recording_retention_policy_attached": False}, "effect": {"decision": "deny", "reason": "recording_retention_required", "required_action": "attach_recording_retention"}},
	{"name": "export_requires_approval", "description": "Workspace exports require approval.", "condition": {"operation": "export_workspace", "export_approved": False}, "effect": {"decision": "deny", "reason": "workspace_export_approval_required", "required_action": "approve_workspace_export"}},
	{"name": "ai_agent_requires_registration", "description": "AI collaborators must be registered.", "condition": {"ai_agent_participant": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "ai_agent_registration_required", "required_action": "register_ai_agent"}},
	{"name": "ai_agent_requires_scope", "description": "AI collaborators require explicit workspace scope.", "condition": {"ai_agent_participant": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "ai_agent_scope_required", "required_action": "set_agent_scope"}},
	{"name": "ai_contribution_requires_disclosure", "description": "AI collaborator contributions require disclosure.", "condition": {"ai_agent_participant": True, "ai_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ai_contribution_disclosure_required", "required_action": "disclose_ai_contribution"}},
	{"name": "collaboration_agent_runtime_supported", "description": "Collaboration agents must use supported provider-neutral runtimes.", "condition": {"operation": "register_collaboration_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_collaboration_agent_runtime", "required_action": "choose_supported_collaboration_agent_runtime"}},
	{"name": "collaboration_agent_role_supported", "description": "Collaboration agents must use supported collaboration-governance roles.", "condition": {"operation": "register_collaboration_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_collaboration_agent_role", "required_action": "choose_supported_collaboration_agent_role"}},
	{"name": "collaboration_agent_requires_scope", "description": "Collaboration agents require explicit workspace, session, artifact, annotation, decision, presence, protocol, guest, or lifecycle scope.", "condition": {"operation": "register_collaboration_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "collaboration_agent_scope_required", "required_action": "declare_collaboration_agent_scope"}},
	{"name": "collaboration_agent_requires_owner", "description": "Collaboration agents require an accountable owner.", "condition": {"operation": "register_collaboration_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "collaboration_agent_owner_required", "required_action": "assign_collaboration_agent_owner"}},
	{"name": "collaboration_agent_requires_purpose", "description": "Collaboration agents require a documented collaboration-governance purpose.", "condition": {"operation": "register_collaboration_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "collaboration_agent_purpose_required", "required_action": "document_collaboration_agent_purpose"}},
	{"name": "collaboration_agent_requires_contribution_disclosure", "description": "Collaboration agents must disclose machine-authored workspace, session, artifact, decision, protocol, and lifecycle contributions.", "condition": {"operation": "register_collaboration_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "collaboration_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "collaboration_agent_privileged_role_requires_human_approval", "description": "Privileged collaboration-agent roles require human approval evidence.", "condition": {"operation": "register_collaboration_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "collaboration_agent_human_approval_required", "required_action": "record_human_collaboration_agent_approval"}},
	{"name": "colb_lifecycle_batch_requires_mutations", "description": "COLB lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_colb_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "colb_lifecycle_batch_empty", "required_action": "include_colb_lifecycle_mutations"}},
	{"name": "bytewax_colb_stream_required", "description": "COLB lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_colb_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_colb_lifecycle_batch_to_bytewax"}},
	{"name": "duplicate_artifact_id_blocked", "description": "Duplicate artifact IDs are blocked within a tenant.", "condition": {"operation": "share_artifact", "duplicate_artifact_id": True}, "effect": {"decision": "deny", "reason": "duplicate_artifact_id", "required_action": "reuse_existing_artifact"}},
	{"name": "collaboration_state_change_requires_audit", "description": "Collaboration state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "collaboration_audit_event_required", "required_action": "record_collaboration_audit"}},
	{"name": "cross_tenant_collaboration_access_denied", "description": "Collaboration records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_collaboration_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_collaboration_mutation_requires_bytewax", "description": "Batch collaboration mutations must use Bytewax event streams.", "condition": {"operation": "batch_collaboration_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/colb/dashboard", "component": "COLBDashboard", "permission": "colb:view", "nav_group": "Overview"},
	{"name": "workspaces", "path": "/colb/workspaces", "component": "WorkspaceManager", "permission": "colb:create_workspace", "nav_group": "Workspaces"},
	{"name": "sessions", "path": "/colb/sessions", "component": "SessionConsole", "permission": "colb:manage_sessions", "nav_group": "Realtime"},
	{"name": "presence", "path": "/colb/presence", "component": "PresenceSync", "permission": "colb:view", "nav_group": "Realtime"},
	{"name": "artifacts", "path": "/colb/artifacts", "component": "ArtifactBoard", "permission": "colb:collaborate", "nav_group": "Artifacts"},
	{"name": "annotations", "path": "/colb/annotations", "component": "AnnotationThreads", "permission": "colb:collaborate", "nav_group": "Artifacts"},
	{"name": "decisions", "path": "/colb/decisions", "component": "DecisionLog", "permission": "colb:collaborate", "nav_group": "Artifacts"},
	{"name": "agents", "path": "/colb/agents", "component": "CollaborationAgentRoster", "permission": "colb:manage_sessions", "nav_group": "Realtime"},
	{"name": "lifecycle", "path": "/colb/lifecycle", "component": "COLBLifecycleBatchMonitor", "permission": "colb:admin", "nav_group": "Operations"},
	{"name": "protocols", "path": "/colb/protocols", "component": "ProtocolMonitor", "permission": "colb:admin", "nav_group": "Operations"},
	{"name": "analytics", "path": "/colb/analytics", "component": "CollaborationAnalytics", "permission": "colb:view", "nav_group": "Operations"},
	{"name": "audit", "path": "/colb/audit", "component": "CollaborationAuditTrail", "permission": "colb:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/colb/settings", "component": "COLBSettings", "permission": "colb:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "colb_collaboration_workspace",
	"tokens": {
		"color.primary": "#2B6CB0",
		"color.accent": "#DD6B20",
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
		"workspace_grid": {"icon": "users", "status_indicator": "workspace-pill", "risk_style": "membership-band"},
		"session_canvas": {"visual": "collaborative-surface", "highlight": "presence-chip"},
		"artifact_board": {"visual": "kanban-board", "status_style": "artifact-chip"},
		"annotation_panel": {"visual": "threaded-comments", "status_style": "decision-chip"},
		"decision_log": {"visual": "decision-table", "status_style": "evidence-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"collaboration_agent_roster": {"icon": "bot", "visual": "agent-roster", "status_style": "scope-chip"},
		"bytewax_lifecycle_panel": {"icon": "activity", "visual": "lifecycle-batch-list", "status_style": "stream-chip"},
		"protocol_monitor": {"visual": "protocol-health-table", "status_style": "transport-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "collaboration-chip"},
	},
}


STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"stream": "apg.colb.lifecycle",
	"key": "tenant_id",
	"events": [
		"workspace_created",
		"workspace_updated",
		"workspace_archived",
		"session_started",
		"session_ended",
		"artifact_created",
		"artifact_updated",
		"artifact_shared",
		"annotation_added",
		"decision_recorded",
		"participant_added",
		"participant_removed",
		"agent_registered",
	],
	"guardrails": [
		"colb_batch_requires_bytewax",
		"colb_privileged_action_requires_human_approval",
	],
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable COLB capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "colb",
		"display_name": "Collaboration Tools",
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/colb/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"agents": agent_manifest(config),
		"streaming": deepcopy(STREAMING),
	}


def agent_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return provider-neutral collaboration agent composition metadata."""
	agents = (config or DEFAULT_CONFIGURATION)["agents"]
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


def streaming_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return Bytewax lifecycle stream metadata for COLB composition."""
	streaming = (config or DEFAULT_CONFIGURATION)["streaming"]
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
	"""Evaluate default COLB governance rules."""
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
