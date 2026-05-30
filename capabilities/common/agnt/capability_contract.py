"""Executable capability contract for APG AI Agent Composition."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_AGENT_RUNTIMES = ["local", "codex", "claude_code", "opencode", "pi"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"agents": {"model_required": True, "runtime_required": True, "system_prompt_required": True, "tool_allowlist_required": True, "io_contract_required": True},
	"teams": {"minimum_agents": 1, "handoff_validation_required": True, "cycle_review_required": True, "parallel_execution_enabled": True},
	"runtimes": {"default_runtime": "local", "registered": SUPPORTED_AGENT_RUNTIMES, "external_runtime_approval_required": True, "workspace_sandbox_required": True, "runtime_requester_required": True, "cost_limit_required": True},
	"memory": {"vector_memory_supported": True, "memory_store_required": True, "retention_policy_required": True, "sensitive_memory_redaction_required": True},
	"governance": {"require_tenant_context": True, "audit_agent_runs": True, "cost_limit_required": True, "human_approval_for_external_side_effects": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "cost_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.AgntService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "ai_orchestration": "aicr", "workflow": "wflo", "audit_sink": "audl", "sandbox": "sbox"},
	"ui": {"enable_agent_registry": True, "enable_team_builder": True, "enable_runtime_manager": True, "enable_execution_trace": True, "enable_approval_queue": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "agnt_agent_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "agents", "teams", "runtimes", "memory", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["agents", "teams", "runtimes", "memory", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All agent-composition operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "agent_requires_model", "description": "First-class AI agents require a model.", "condition": {"operation": "register_agent", "model_present": False}, "effect": {"decision": "deny", "reason": "agent_model_required", "required_action": "declare_agent_model"}},
	{"name": "agent_requires_system_prompt", "description": "First-class AI agents require an explicit system prompt.", "condition": {"operation": "register_agent", "system_prompt_present": False}, "effect": {"decision": "deny", "reason": "agent_system_prompt_required", "required_action": "declare_agent_system_prompt"}},
	{"name": "agent_requires_tool_allowlist", "description": "First-class AI agents require an explicit tool allowlist.", "condition": {"operation": "register_agent", "tool_allowlist_present": False}, "effect": {"decision": "deny", "reason": "agent_tool_allowlist_required", "required_action": "declare_agent_tool_allowlist"}},
	{"name": "agent_requires_io_contract", "description": "First-class AI agents require input and output contracts.", "condition": {"operation": "register_agent", "io_contract_present": False}, "effect": {"decision": "deny", "reason": "agent_io_contract_required", "required_action": "declare_agent_io_contract"}},
	{"name": "agent_requires_memory_policy", "description": "First-class AI agents require memory policy.", "condition": {"operation": "register_agent", "memory_policy_present": False}, "effect": {"decision": "deny", "reason": "agent_memory_policy_required", "required_action": "declare_agent_memory_policy"}},
	{"name": "agent_runtime_must_be_registered", "description": "Agent runtimes must resolve to a registered backend.", "condition": {"runtime_registered": False}, "effect": {"decision": "deny", "reason": "runtime_not_registered", "required_action": "register_runtime_backend"}},
	{"name": "runtime_requires_cost_limit", "description": "Agent runtimes require cost limits.", "condition": {"operation": "register_runtime", "cost_limit_present": False}, "effect": {"decision": "deny", "reason": "runtime_cost_limit_required", "required_action": "declare_runtime_cost_limit"}},
	{"name": "runtime_approval_requires_requester", "description": "External runtime approval requests require requester identity.", "condition": {"operation": "request_runtime_approval", "runtime_requester_present": False}, "effect": {"decision": "deny", "reason": "runtime_requester_required", "required_action": "declare_runtime_requester"}},
	{"name": "runtime_approval_decision_requires_reviewer", "description": "Runtime approval decisions require reviewer identity.", "condition": {"operation": "decide_runtime_approval", "reviewer_present": False}, "effect": {"decision": "deny", "reason": "runtime_approval_reviewer_required", "required_action": "declare_runtime_reviewer"}},
	{"name": "runtime_approval_decision_requires_notes", "description": "Runtime approval decisions require notes.", "condition": {"operation": "decide_runtime_approval", "decision_notes_present": False}, "effect": {"decision": "deny", "reason": "runtime_approval_notes_required", "required_action": "record_runtime_approval_notes"}},
	{"name": "team_requires_agent", "description": "Agent teams require at least one member.", "condition": {"operation": "register_team", "agent_count_lt": 1}, "effect": {"decision": "deny", "reason": "team_agent_required", "required_action": "add_team_agent"}},
	{"name": "handoff_endpoint_must_resolve", "description": "Handoff graph endpoints must resolve to declared agents.", "condition": {"handoff_endpoint_resolved": False}, "effect": {"decision": "deny", "reason": "handoff_endpoint_unknown", "required_action": "fix_handoff_reference"}},
	{"name": "workspace_runtime_requires_sandbox", "description": "Workspace-aware agent runtimes require sandbox policy.", "condition": {"workspace_runtime": True, "sandbox_policy_attached": False}, "effect": {"decision": "deny", "reason": "workspace_sandbox_required", "required_action": "attach_sandbox_policy"}},
	{"name": "external_runtime_requires_approval", "description": "External agent backends require approval before use.", "condition": {"external_runtime": True, "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "external_runtime_approval_required", "required_action": "review_external_runtime"}},
	{"name": "execution_plan_requires_objective", "description": "Agent execution plans require an objective.", "condition": {"operation": "plan_execution", "objective_present": False}, "effect": {"decision": "deny", "reason": "execution_objective_required", "required_action": "declare_execution_objective"}},
	{"name": "agnt_state_change_requires_audit", "description": "Agent composition lifecycle changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "agnt_audit_event_required", "required_action": "record_agent_audit_event"}},
	{"name": "cross_tenant_agent_access_denied", "description": "Agent composition records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_agent_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_agent_mutation_requires_bytewax", "description": "Batch agent mutations must use Bytewax event streams.", "condition": {"operation": "batch_agent_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/agnt/dashboard", "component": "AGNTDashboard", "permission": "agnt:view", "nav_group": "Overview"},
	{"name": "agents", "path": "/agnt/agents", "component": "AgentRegistry", "permission": "agnt:compose", "nav_group": "Agents"},
	{"name": "teams", "path": "/agnt/teams", "component": "AgentTeamBuilder", "permission": "agnt:compose", "nav_group": "Teams"},
	{"name": "handoffs", "path": "/agnt/handoffs", "component": "HandoffGraph", "permission": "agnt:compose", "nav_group": "Teams"},
	{"name": "runtimes", "path": "/agnt/runtimes", "component": "RuntimeManager", "permission": "agnt:manage_runtimes", "nav_group": "Runtimes"},
	{"name": "executions", "path": "/agnt/executions", "component": "ExecutionTrace", "permission": "agnt:run", "nav_group": "Operations"},
	{"name": "memory", "path": "/agnt/memory", "component": "AgentMemoryPolicy", "permission": "agnt:admin", "nav_group": "Governance"},
	{"name": "approvals", "path": "/agnt/approvals", "component": "RuntimeApprovalQueue", "permission": "agnt:manage_runtimes", "nav_group": "Governance"},
	{"name": "audit", "path": "/agnt/audit", "component": "AgentAuditTrail", "permission": "agnt:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/agnt/analytics", "component": "AgentAnalytics", "permission": "agnt:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/agnt/settings", "component": "AGNTSettings", "permission": "agnt:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "agnt_agent_ops",
	"tokens": {"color.primary": "#2B4C7E", "color.accent": "#2F855A", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"agent_card": {"icon": "bot", "status_indicator": "runtime-pill", "risk_style": "approval-band"}, "team_graph": {"visual": "handoff-dag", "highlight": "edge-chip"}, "runtime_matrix": {"visual": "backend-table", "status_style": "availability-chip"}, "execution_trace": {"visual": "timeline", "status_style": "decision-chip"}, "approval_queue": {"icon": "shield-check", "status_style": "decision-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.agnt.lifecycle",
	"state": ["runtimes", "runtime_approvals", "agents", "teams", "execution_plans", "audit_events"],
	"events": ["runtime_approval_requested", "runtime_approval_decided", "runtime_registered", "agent_registered", "team_registered", "execution_plan_built"],
	"batch_mutation_guardrail": "batch_agent_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "agnt", "display_name": "AI Agent Composition", "provides": ["agent_registry", "runtime_registry", "agent_teams", "handoff_graphs", "execution_plans", "runtime_approval_governance"], "requires": ["aicr", "sbox", "audl"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/agnt/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
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
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
