"""Executable capability contract for APG Custom Scripting Engine."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_SCPT_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_SCPT_AGENT_ROLES: list[str] = [
	"author",
	"reviewer",
	"policy_advisor",
	"test_generator",
	"runtime_triage",
	"lifecycle_batch_reviewer",
	"script_steward",
]

PRIVILEGED_SCPT_AGENT_ROLES: list[str] = [
	"author",
	"reviewer",
	"policy_advisor",
	"runtime_triage",
	"lifecycle_batch_reviewer",
	"script_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"scripts": {
		"script_owner_required": True,
		"script_name_required": True,
		"script_source_required": True,
		"versioning_enabled": True,
		"review_required_for_publish": True,
		"package_policy_required": True,
		"sandbox_policy_required": True,
		"workflow_binding_policy_required": True,
		"retirement_reason_required": True,
		"allowed_languages": ["python", "javascript", "apg"],
	},
	"sandbox": {
		"sandbox_owner_required": True,
		"sandbox_required": True,
		"network_disabled_by_default": True,
		"max_runtime_seconds": 300,
		"max_memory_mb": 512,
		"supported_isolation_modes": ["process", "container", "wasm"],
		"health_check_required": True,
		"block_reason_required": True,
		"retirement_reason_required": True,
	},
	"packages": {
		"package_policy_owner_required": True,
		"allowlist_required": True,
		"secret_access_policy_required": True,
		"filesystem_access_policy_required": True,
		"network_access_policy_required": True,
		"dangerous_import_blocking": True,
		"approval_required_for_host_access": True,
	},
	"executions": {
		"requested_by_required": True,
		"published_script_required": True,
		"ready_sandbox_required": True,
		"event_stream": "bytewax",
		"completion_evidence_required": True,
		"cancel_reason_required": True,
		"runtime_metrics_required": True,
	},
	"scripting_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_SCPT_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_SCPT_AGENT_ROLES,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_SCPT_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_SCPT_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_SCPT_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_scpt_agent_adapter",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_executions": True,
		"dangerous_permission_approval_required": True,
		"workflow_binding_policy_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"execution_metrics_required": True,
		"sandbox_metrics_required": True,
		"event_stream": "bytewax",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "scpt.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"package_policy_batch",
			"sandbox_batch",
			"script_batch",
			"approval_batch",
			"execution_batch",
			"scripting_agent_batch",
			"audit_batch",
		],
		"topics": [
			"scpt.packages",
			"scpt.sandboxes",
			"scpt.scripts",
			"scpt.approvals",
			"scpt.executions",
			"scpt.agents",
			"scpt.audit",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "service.ScptService",
		"runtime_helpers": "script_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"workflow": "wflo",
		"security": "secu",
		"identity": "auth",
		"audit_sink": "audl",
		"scheduler": "schd",
		"no_code_builder": "ncod",
		"ai_core": "aicr",
		"agent_adapter": "aicr_provider_neutral_scpt_agent_adapter",
		"monitoring": "moni",
		"theme": "them",
	},
	"ui": {
		"enable_script_workbench": True,
		"enable_execution_console": True,
		"enable_sandbox_monitor": True,
		"enable_package_policy": True,
		"enable_agent_panel": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "scpt_script_workbench",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"scripts",
		"sandbox",
		"packages",
		"executions",
		"scripting_agents",
		"agents",
		"governance",
		"observability",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"scripts",
		"sandbox",
		"packages",
		"executions",
		"scripting_agents",
		"agents",
		"governance",
		"observability",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All scripting operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "script_requires_owner", "description": "Scripts require an accountable owner.", "condition": {"operation": "create_script", "script_owner_assigned": False}, "effect": {"decision": "deny", "reason": "script_owner_required", "required_action": "assign_script_owner"}},
	{"name": "script_requires_name", "description": "Scripts require a readable name.", "condition": {"operation": "create_script", "script_name_present": False}, "effect": {"decision": "deny", "reason": "script_name_required", "required_action": "name_script"}},
	{"name": "script_requires_source", "description": "Scripts require source text.", "condition": {"operation": "create_script", "script_source_present": False}, "effect": {"decision": "deny", "reason": "script_source_required", "required_action": "add_script_source"}},
	{"name": "script_requires_package_policy", "description": "Scripts require an attached package policy.", "condition": {"operation": "create_script", "package_policy_attached": False}, "effect": {"decision": "deny", "reason": "package_policy_required", "required_action": "attach_package_policy"}},
	{"name": "script_requires_sandbox_policy", "description": "Scripts require an attached sandbox policy.", "condition": {"operation": "create_script", "sandbox_attached": False}, "effect": {"decision": "deny", "reason": "sandbox_required", "required_action": "attach_sandbox"}},
	{"name": "script_blocked_import_denied", "description": "Scripts may not import blocked modules.", "condition": {"operation": "create_script", "blocked_import_detected": True}, "effect": {"decision": "deny", "reason": "blocked_import_denied", "required_action": "remove_blocked_import"}},
	{"name": "sandbox_required", "description": "Script execution requires an active sandbox.", "condition": {"operation": "execute_script", "sandbox_attached": False}, "effect": {"decision": "deny", "reason": "sandbox_required", "required_action": "attach_sandbox"}},
	{"name": "sandbox_requires_owner", "description": "Sandboxes require an accountable owner.", "condition": {"operation": "create_sandbox", "sandbox_owner_assigned": False}, "effect": {"decision": "deny", "reason": "sandbox_owner_required", "required_action": "assign_sandbox_owner"}},
	{"name": "sandbox_limits_must_be_positive", "description": "Sandbox resource limits must be positive.", "condition": {"operation": "create_sandbox", "sandbox_limits_positive": False}, "effect": {"decision": "deny", "reason": "sandbox_resource_limits_must_be_positive", "required_action": "set_positive_sandbox_limits"}},
	{"name": "sandbox_requires_health_check", "description": "Sandboxes require health-check evidence.", "condition": {"operation": "create_sandbox", "health_check_attached": False}, "effect": {"decision": "deny", "reason": "sandbox_health_check_required", "required_action": "attach_sandbox_health_check"}},
	{"name": "sandbox_block_requires_reason", "description": "Blocking a sandbox requires a reason.", "condition": {"operation": "change_sandbox_state", "target_sandbox_state": "blocked", "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "sandbox_block_reason_required", "required_action": "record_sandbox_block_reason"}},
	{"name": "sandbox_retirement_requires_reason", "description": "Retiring a sandbox requires a reason.", "condition": {"operation": "change_sandbox_state", "target_sandbox_state": "retired", "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "sandbox_retirement_reason_required", "required_action": "record_sandbox_retirement_reason"}},
	{"name": "package_policy_requires_owner", "description": "Package policies require an accountable owner.", "condition": {"operation": "create_package_policy", "package_policy_owner_assigned": False}, "effect": {"decision": "deny", "reason": "package_policy_owner_required", "required_action": "assign_package_policy_owner"}},
	{"name": "package_allowlist_required", "description": "Package policies require an allowlist.", "condition": {"operation": "create_package_policy", "package_allowlist_present": False}, "effect": {"decision": "deny", "reason": "package_allowlist_required", "required_action": "define_package_allowlist"}},
	{"name": "secret_access_requires_approval", "description": "Secret access requires approval.", "condition": {"operation": "create_package_policy", "secret_access_requested": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "secret_access_policy_required", "required_action": "record_secret_access_approval"}},
	{"name": "filesystem_access_requires_approval", "description": "Filesystem access requires approval.", "condition": {"operation": "create_package_policy", "filesystem_access_requested": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "filesystem_access_policy_required", "required_action": "record_filesystem_access_approval"}},
	{"name": "dangerous_permission_requires_approval", "description": "Dangerous permissions require approval.", "condition": {"dangerous_permission_requested": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "dangerous_permission_approval_required", "required_action": "record_permission_approval"}},
	{"name": "external_network_requires_policy", "description": "Network access requires an explicit policy.", "condition": {"network_access_requested": True, "network_policy_attached": False}, "effect": {"decision": "deny", "reason": "network_policy_required", "required_action": "attach_network_policy"}},
	{"name": "high_resource_script_requires_review", "description": "High resource scripts require review.", "condition": {"requested_memory_mb_gt": 512, "resource_review_recorded": False}, "effect": {"decision": "require_review", "reason": "resource_review_required", "required_action": "review_script_resources"}},
	{"name": "publish_requires_review", "description": "Script publication requires review evidence.", "condition": {"operation": "publish_script", "script_reviewed": False}, "effect": {"decision": "deny", "reason": "script_review_required", "required_action": "review_script"}},
	{"name": "publish_requires_approval_for_dangerous_permissions", "description": "Publishing scripts with dangerous permissions requires approval.", "condition": {"operation": "publish_script", "dangerous_permission_requested": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "dangerous_permission_approval_required", "required_action": "approve_dangerous_permissions"}},
	{"name": "workflow_binding_requires_policy", "description": "Workflow bindings require a policy reference.", "condition": {"operation": "bind_workflow", "workflow_binding_policy_attached": False}, "effect": {"decision": "deny", "reason": "workflow_binding_policy_required", "required_action": "attach_workflow_binding_policy"}},
	{"name": "execution_requires_published_script", "description": "Execution requires a published script.", "condition": {"operation": "execute_script", "script_published": False}, "effect": {"decision": "deny", "reason": "published_script_required", "required_action": "publish_script"}},
	{"name": "execution_requires_ready_sandbox", "description": "Execution requires a ready sandbox.", "condition": {"operation": "execute_script", "sandbox_ready": False}, "effect": {"decision": "deny", "reason": "sandbox_not_ready", "required_action": "restore_sandbox"}},
	{"name": "execution_requested_by_required", "description": "Execution requires a requesting actor.", "condition": {"operation": "execute_script", "requested_by_present": False}, "effect": {"decision": "deny", "reason": "requested_by_required", "required_action": "record_requesting_actor"}},
	{"name": "execution_requires_bytewax_stream", "description": "Script execution events must use Bytewax streams.", "condition": {"operation": "execute_script", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "execution_completion_requires_evidence", "description": "Execution completion requires evidence.", "condition": {"operation": "complete_execution", "completion_evidence_present": False}, "effect": {"decision": "deny", "reason": "execution_completion_evidence_required", "required_action": "record_completion_evidence"}},
	{"name": "execution_metrics_must_be_non_negative", "description": "Execution metrics must be non-negative.", "condition": {"operation": "complete_execution", "execution_metrics_valid": False}, "effect": {"decision": "deny", "reason": "execution_metrics_must_be_non_negative", "required_action": "correct_execution_metrics"}},
	{"name": "execution_cancel_requires_reason", "description": "Execution cancellation requires a reason.", "condition": {"operation": "cancel_execution", "cancel_reason_present": False}, "effect": {"decision": "deny", "reason": "execution_cancel_reason_required", "required_action": "record_cancel_reason"}},
	{"name": "script_retirement_requires_reason", "description": "Script retirement requires a reason.", "condition": {"operation": "retire_script", "retirement_reason_present": False}, "effect": {"decision": "deny", "reason": "script_retirement_reason_required", "required_action": "record_retirement_reason"}},
	{"name": "scripting_agent_requires_id", "description": "First-class scripting agents require stable identifiers.", "condition": {"operation": "register_scripting_agent", "agent_id_present": False}, "effect": {"decision": "deny", "reason": "scripting_agent_id_required", "required_action": "assign_scripting_agent_id"}},
	{"name": "scripting_agent_requires_name", "description": "First-class scripting agents require readable names.", "condition": {"operation": "register_scripting_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "scripting_agent_name_required", "required_action": "name_scripting_agent"}},
	{"name": "scripting_agent_runtime_supported", "description": "First-class scripting agents must use a configured provider-neutral runtime.", "condition": {"operation": "register_scripting_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "scripting_agent_runtime_not_supported", "required_action": "choose_supported_scripting_agent_runtime"}},
	{"name": "scripting_agent_role_supported", "description": "First-class scripting agents must use supported scripting-governance roles.", "condition": {"operation": "register_scripting_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "scripting_agent_role_not_supported", "required_action": "choose_supported_scripting_agent_role"}},
	{"name": "scripting_agent_requires_scope", "description": "First-class scripting agents require script, sandbox, package, execution, approval, or lifecycle scope.", "condition": {"operation": "register_scripting_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "scripting_agent_scope_required", "required_action": "set_scripting_agent_scope"}},
	{"name": "scripting_agent_requires_owner", "description": "First-class scripting agents require an accountable owner.", "condition": {"operation": "register_scripting_agent", "agent_owner_present": False}, "effect": {"decision": "deny", "reason": "scripting_agent_owner_required", "required_action": "assign_scripting_agent_owner"}},
	{"name": "scripting_agent_requires_purpose", "description": "First-class scripting agents require a documented scripting-governance purpose.", "condition": {"operation": "register_scripting_agent", "agent_purpose_present": False}, "effect": {"decision": "deny", "reason": "scripting_agent_purpose_required", "required_action": "document_scripting_agent_purpose"}},
	{"name": "scripting_agent_requires_disclosure", "description": "First-class scripting agent contributions require visible machine-contribution disclosure.", "condition": {"operation": "register_scripting_agent", "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "scripting_agent_disclosure_required", "required_action": "disclose_scripting_agent"}},
	{"name": "scripting_agent_privileged_role_requires_human_approval", "description": "Privileged scripting-agent roles require human approval evidence.", "condition": {"operation": "register_scripting_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "scripting_agent_human_approval_required", "required_action": "record_scripting_agent_human_approval"}},
	{"name": "scpt_lifecycle_batch_requires_mutations", "description": "SCPT lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_scpt_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "scpt_lifecycle_batch_empty", "required_action": "include_scpt_lifecycle_mutations"}},
	{"name": "scpt_lifecycle_operation_supported", "description": "SCPT lifecycle batches must use configured lifecycle operations.", "condition": {"operation": "validate_scpt_lifecycle_batch", "lifecycle_operation_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_scpt_lifecycle_operation", "required_action": "choose_supported_scpt_lifecycle_operation"}},
	{"name": "bytewax_scpt_lifecycle_stream_required", "description": "SCPT lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_scpt_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_scpt_lifecycle_batch_to_bytewax"}},
	{"name": "script_state_change_requires_audit", "description": "Script, sandbox, and execution state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "scripting_audit_event_required", "required_action": "record_scripting_audit_event"}},
	{"name": "cross_tenant_script_access_denied", "description": "Scripting records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_script_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_script_mutation_requires_bytewax", "description": "Batch script mutations must use Bytewax event streams.", "condition": {"operation": "batch_script_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/scpt/dashboard", "component": "SCPTDashboard", "permission": "scpt:view", "nav_group": "Overview"},
	{"name": "workbench", "path": "/scpt/workbench", "component": "ScriptWorkbench", "permission": "scpt:write", "nav_group": "Scripts"},
	{"name": "scripts", "path": "/scpt/scripts", "component": "ScriptRegistry", "permission": "scpt:view", "nav_group": "Scripts"},
	{"name": "executions", "path": "/scpt/executions", "component": "ExecutionConsole", "permission": "scpt:execute", "nav_group": "Runtime"},
	{"name": "sandboxes", "path": "/scpt/sandboxes", "component": "SandboxMonitor", "permission": "scpt:admin", "nav_group": "Runtime"},
	{"name": "packages", "path": "/scpt/packages", "component": "PackagePolicy", "permission": "scpt:approve", "nav_group": "Governance"},
	{"name": "approvals", "path": "/scpt/approvals", "component": "ScriptApprovals", "permission": "scpt:approve", "nav_group": "Governance"},
	{"name": "agents", "path": "/scpt/agents", "component": "ScriptingAgentPanel", "permission": "scpt:write", "nav_group": "Scripts"},
	{"name": "lifecycle", "path": "/scpt/lifecycle", "component": "SCPTLifecycleBatchMonitor", "permission": "scpt:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/scpt/audit", "component": "ScriptAuditTrail", "permission": "scpt:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/scpt/analytics", "component": "ScriptAnalytics", "permission": "scpt:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/scpt/settings", "component": "SCPTSettings", "permission": "scpt:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "scpt_script_workbench",
	"tokens": {
		"color.primary": "#2A4365",
		"color.accent": "#805AD5",
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
		"script_editor": {"icon": "code-2", "status_indicator": "script-pill", "risk_style": "permission-band"},
		"execution_log": {"visual": "log-stream", "highlight": "runtime-chip"},
		"sandbox_monitor": {"visual": "resource-meter", "status_style": "isolation-chip"},
		"package_policy": {"visual": "allowlist-table", "status_style": "approval-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"scripting_agent_roster": {"visual": "agent-roster", "status_style": "approval-chip"},
		"bytewax_lifecycle_panel": {"visual": "stream-batch-monitor", "status_style": "processor-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "script-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.scpt.lifecycle",
	"state": ["package_policies", "sandboxes", "scripts", "approvals", "executions", "scripting_agents"],
	"events": [
		"package_policy_created",
		"sandbox_created",
		"sandbox_state_changed",
		"script_created",
		"script_reviewed",
		"script_approved",
		"script_published",
		"script_retired",
		"workflow_bound",
		"script_execution_started",
		"script_execution_completed",
		"script_execution_cancelled",
		"scripting_agent_registered",
	],
	"batch_mutation_guardrail": "batch_script_mutation_requires_bytewax",
	"engine": "bytewax",
	"lifecycle_stream": "scpt.lifecycle",
	"watermark": "event_time",
	"required_processor": "bytewax",
	"required_operations": DEFAULT_CONFIGURATION["streaming"]["required_operations"],
	"topics": DEFAULT_CONFIGURATION["streaming"]["topics"],
	"broker_core_dependency_allowed": False,
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SCPT capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "scpt",
		"display_name": "Custom Scripting Engine",
		"provides": ["script_registry", "secure_sandbox", "workflow_extensions", "package_policy", "script_execution", "scripting_agent_composition", "script_governance", "bytewax_script_lifecycle"],
		"requires": ["wflo", "secu", "auth", "audl", "aicr"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/scpt/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"agents": agent_manifest(config),
		"streaming": streaming_manifest(config),
	}


def agent_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return first-class provider-neutral scripting-agent composition metadata."""
	config = config or DEFAULT_CONFIGURATION
	return deepcopy(config["agents"])


def streaming_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return Bytewax lifecycle metadata for scripting composition state."""
	config = config or DEFAULT_CONFIGURATION
	streaming = deepcopy(STREAMING)
	streaming.update(deepcopy(config["streaming"]))
	return streaming


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default SCPT governance rules."""
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
