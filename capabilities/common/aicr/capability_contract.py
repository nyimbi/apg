"""Executable capability contract for APG AI Core Framework."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_AICR_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AICR_AGENT_ROLES = [
	"model_reviewer",
	"prompt_reviewer",
	"inference_reviewer",
	"safety_reviewer",
	"evaluation_reviewer",
	"routing_reviewer",
	"tool_reviewer",
	"cost_reviewer",
	"model_steward",
]
PRIVILEGED_AICR_AGENT_ROLES = [
	"model_reviewer",
	"prompt_reviewer",
	"inference_reviewer",
	"safety_reviewer",
	"evaluation_reviewer",
	"routing_reviewer",
	"tool_reviewer",
	"cost_reviewer",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"services": {
		"registry_enabled": True,
		"service_owner_required": True,
		"endpoint_health_required": True,
		"max_services_per_tenant": 100,
		"credential_vault_required": True,
	},
	"providers": {
		"supported_provider_types": ["local", "ollama", "openai", "anthropic", "codex", "claude_code", "opencode", "pi", "http"],
		"provider_policy_required": True,
		"credential_vault_required": True,
		"egress_policy_required": True,
	},
	"models": {
		"model_owner_required": True,
		"model_policy_required": True,
		"evaluation_required_before_promotion": True,
		"retirement_impact_review_required": True,
		"supported_modalities": ["text", "image", "audio", "video", "tabular", "multimodal"],
	},
	"inference": {
		"default_timeout_seconds": 60,
		"max_concurrent_requests": 10000,
		"max_context_tokens_without_review": 128000,
		"model_policy_required": True,
		"prompt_audit_enabled": True,
		"tool_allowlist_required": True,
		"pii_redaction_required": True,
	},
	"workflows": {
		"workflow_orchestration_enabled": True,
		"multi_modal_fusion_enabled": True,
		"human_approval_for_high_risk": True,
		"minimum_steps": 1,
		"owner_required": True,
	},
	"agent_runtimes": {
		"first_class_agents_enabled": True,
		"supported_runtimes": ["codex", "claude_code", "opencode", "pi", "ollama", "custom_http"],
		"tool_policy_required": True,
		"handoff_audit_required": True,
		"human_approval_for_external_actions": True,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_AICR_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_AICR_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_AICR_AGENT_ROLES,
		"require_owner": True,
		"require_purpose": True,
		"require_scope": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "provider_neutral_cli_or_api_adapter",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "aicr.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"prompt_batch",
			"inference_batch",
			"evaluation_batch",
			"safety_batch",
			"routing_batch",
			"ai_agent_batch",
		],
		"topics": [
			"aicr.models",
			"aicr.prompts",
			"aicr.inference",
			"aicr.evaluations",
			"aicr.safety",
			"aicr.routing",
			"aicr.agents",
		],
		"broker_core_dependency_allowed": False,
	},
	"governance": {
		"require_tenant_context": True,
		"auth_required": True,
		"monitoring_required": True,
		"ai_audit_events_required": True,
		"cross_tenant_routing_allowed": False,
	},
	"observability": {
		"metrics_required": True,
		"trace_required": True,
		"cost_tracking_required": True,
		"drift_monitoring_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.AicrService",
		"production_runtime": "service.AICoreService",
		"http_api": "api.app",
		"event_stream": "bytewax",
		"auth_provider": "auth",
		"audit_sink": "audl",
		"metrics_sink": "moni",
		"credential_vault": "keym",
		"model_lifecycle": "mlcm",
		"agent_composition": "agnt",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_service_registry": True,
		"enable_provider_registry": True,
		"enable_model_catalog": True,
		"enable_inference_console": True,
		"enable_workflow_designer": True,
		"enable_agent_runtime_console": True,
		"enable_ai_agent_roster": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_governance_center": True,
		"enable_evaluation_console": True,
		"enable_cost_and_metrics": True,
		"enable_audit_timeline": True,
		"enable_settings": True,
	},
	"theme": {
		"default_theme": "aicr_ai_control_console",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"services",
		"providers",
		"models",
		"inference",
		"workflows",
		"agent_runtimes",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"services",
		"providers",
		"models",
		"inference",
		"workflows",
		"agent_runtimes",
		"agents",
		"streaming",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	]} | {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All AI core operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "service_registration_requires_owner", "description": "AI service registration requires an owner.", "condition": {"operation": "register_service", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "service_owner_required", "required_action": "assign_service_owner"}},
	{"name": "service_registration_requires_endpoint", "description": "AI service registration requires a routeable endpoint.", "condition": {"operation": "register_service", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "service_endpoint_required", "required_action": "attach_service_endpoint"}},
	{"name": "provider_type_must_be_supported", "description": "AI providers require a supported provider type.", "condition": {"operation": "register_provider", "provider_type_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_provider_type", "required_action": "choose_supported_provider_type"}},
	{"name": "provider_requires_owner", "description": "AI providers require an accountable owner.", "condition": {"operation": "register_provider", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "provider_owner_required", "required_action": "assign_provider_owner"}},
	{"name": "provider_requires_credential_vault", "description": "External AI providers require credential vault references.", "condition": {"operation": "register_provider", "external_provider": True, "credential_vault_ref_present": False}, "effect": {"decision": "deny", "reason": "provider_credential_vault_required", "required_action": "store_provider_credentials"}},
	{"name": "provider_requires_egress_policy", "description": "External AI providers require egress policy.", "condition": {"operation": "register_provider", "external_provider": True, "egress_policy_attached": False}, "effect": {"decision": "deny", "reason": "provider_egress_policy_required", "required_action": "attach_egress_policy"}},
	{"name": "model_requires_owner", "description": "Model registrations require an owner.", "condition": {"operation": "register_model", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "model_owner_required", "required_action": "assign_model_owner"}},
	{"name": "model_requires_registered_provider", "description": "Models require a registered provider.", "condition": {"operation": "register_model", "provider_registered": False}, "effect": {"decision": "deny", "reason": "registered_provider_required", "required_action": "register_provider"}},
	{"name": "model_requires_policy", "description": "Models require policy metadata.", "condition": {"operation": "register_model", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "model_requires_supported_modality", "description": "Models require supported modality metadata.", "condition": {"operation": "register_model", "modality_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_model_modality", "required_action": "choose_supported_modality"}},
	{"name": "model_promotion_requires_evaluation", "description": "Model promotion requires evaluation evidence.", "condition": {"operation": "promote_model", "evaluation_recorded": False}, "effect": {"decision": "deny", "reason": "model_evaluation_required", "required_action": "record_model_evaluation"}},
	{"name": "model_retirement_requires_impact_review", "description": "Model retirement requires impact review.", "condition": {"operation": "retire_model", "impact_review_recorded": False}, "effect": {"decision": "deny", "reason": "model_retirement_review_required", "required_action": "record_retirement_impact"}},
	{"name": "inference_requires_model_policy", "description": "Inference requires an attached model policy.", "condition": {"operation": "run_inference", "model_policy_attached": False}, "effect": {"decision": "deny", "reason": "model_policy_required", "required_action": "attach_model_policy"}},
	{"name": "unhealthy_service_blocks_routing", "description": "Unhealthy AI services cannot receive routed work.", "condition": {"operation": "run_inference", "service_health": "unhealthy", "routing_requested": True}, "effect": {"decision": "deny", "reason": "service_unhealthy", "required_action": "restore_service_health"}},
	{"name": "large_context_requires_review", "description": "Large context windows require cost and safety review.", "condition": {"operation": "run_inference", "context_tokens_gt": 128000, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "large_context_review_required", "required_action": "record_context_review"}},
	{"name": "high_risk_workflow_requires_approval", "description": "High-risk AI workflows require approval.", "condition": {"operation": "run_inference", "workflow_risk": "high", "approval_recorded": False}, "effect": {"decision": "require_review", "reason": "workflow_approval_required", "required_action": "record_human_approval"}},
	{"name": "pii_inference_requires_redaction", "description": "PII-bearing inference requires redaction policy.", "condition": {"operation": "run_inference", "pii_detected": True, "pii_redaction_enabled": False}, "effect": {"decision": "deny", "reason": "pii_redaction_required", "required_action": "enable_pii_redaction"}},
	{"name": "tool_call_requires_allowlist", "description": "Tool-using inference requires tool allowlist.", "condition": {"operation": "run_inference", "tool_call_requested": True, "tool_allowlist_attached": False}, "effect": {"decision": "deny", "reason": "tool_allowlist_required", "required_action": "attach_tool_allowlist"}},
	{"name": "cross_tenant_routing_denied", "description": "Cross-tenant AI routing is denied by default.", "condition": {"operation": "run_inference", "cross_tenant_route": True}, "effect": {"decision": "deny", "reason": "cross_tenant_routing_denied", "required_action": "use_tenant_scoped_service"}},
	{"name": "cost_limit_requires_review", "description": "Inference above cost limit requires review.", "condition": {"operation": "run_inference", "estimated_cost_gt": 100.0, "cost_review_recorded": False}, "effect": {"decision": "require_review", "reason": "cost_review_required", "required_action": "record_cost_review"}},
	{"name": "workflow_requires_owner", "description": "AI workflows require an owner.", "condition": {"operation": "create_workflow", "owner_assigned": False}, "effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"}},
	{"name": "workflow_requires_steps", "description": "AI workflows require at least one step.", "condition": {"operation": "create_workflow", "steps_present": False}, "effect": {"decision": "deny", "reason": "workflow_steps_required", "required_action": "define_workflow_steps"}},
	{"name": "workflow_requires_registered_services", "description": "AI workflows require registered service bindings.", "condition": {"operation": "create_workflow", "services_registered": False}, "effect": {"decision": "deny", "reason": "workflow_service_bindings_required", "required_action": "bind_registered_services"}},
	{"name": "agent_runtime_must_be_supported", "description": "AI agent runtime must be supported.", "condition": {"operation": "register_agent_runtime", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_agent_runtime", "required_action": "choose_supported_agent_runtime"}},
	{"name": "agent_runtime_requires_tool_policy", "description": "AI agent runtime requires tool policy.", "condition": {"operation": "register_agent_runtime", "tool_policy_attached": False}, "effect": {"decision": "deny", "reason": "agent_tool_policy_required", "required_action": "attach_tool_policy"}},
	{"name": "ai_agent_runtime_supported", "description": "First-class AI agents must use supported runtimes.", "condition": {"operation": "register_ai_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ai_agent_runtime", "required_action": "choose_supported_ai_agent_runtime"}},
	{"name": "ai_agent_role_supported", "description": "First-class AI agents must use supported AI-core roles.", "condition": {"operation": "register_ai_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_ai_agent_role", "required_action": "choose_supported_ai_agent_role"}},
	{"name": "ai_agent_requires_scope", "description": "First-class AI agents require an explicit bounded scope.", "condition": {"operation": "register_ai_agent", "scope_present": False}, "effect": {"decision": "deny", "reason": "ai_agent_scope_required", "required_action": "declare_ai_agent_scope"}},
	{"name": "ai_agent_requires_owner", "description": "First-class AI agents require an accountable owner.", "condition": {"operation": "register_ai_agent", "owner_present": False}, "effect": {"decision": "deny", "reason": "ai_agent_owner_required", "required_action": "assign_ai_agent_owner"}},
	{"name": "ai_agent_requires_purpose", "description": "First-class AI agents require a documented purpose.", "condition": {"operation": "register_ai_agent", "purpose_present": False}, "effect": {"decision": "deny", "reason": "ai_agent_purpose_required", "required_action": "document_ai_agent_purpose"}},
	{"name": "ai_agent_requires_contribution_disclosure", "description": "First-class AI agents must disclose machine-authored AI-core contributions.", "condition": {"operation": "register_ai_agent", "contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "ai_agent_contribution_disclosure_required", "required_action": "disclose_machine_contribution"}},
	{"name": "ai_agent_privileged_role_requires_human_approval", "description": "Privileged AI-agent roles require human approval evidence.", "condition": {"operation": "register_ai_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "ai_agent_human_approval_required", "required_action": "record_human_ai_agent_approval"}},
	{"name": "bytewax_aicr_stream_required", "description": "AICR lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_aicr_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_aicr_lifecycle_batch_to_bytewax"}},
	{"name": "external_agent_action_requires_approval", "description": "External AI agent actions require approval.", "condition": {"operation": "run_agent_action", "external_action": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "external_agent_action_approval_required", "required_action": "record_agent_action_approval"}},
	{"name": "completion_requires_audit_event", "description": "AI completions require audit evidence.", "condition": {"operation": "complete_inference", "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "ai_audit_event_required", "required_action": "record_ai_audit_event"}},
	{"name": "streaming_requires_trace", "description": "Streaming AI output requires trace capture.", "condition": {"operation": "stream_inference", "trace_enabled": False}, "effect": {"decision": "deny", "reason": "ai_trace_required", "required_action": "enable_trace_capture"}},
	{"name": "drift_review_required", "description": "Model drift above threshold requires review.", "condition": {"operation": "record_model_metric", "drift_score_gt": 0.2, "drift_review_recorded": False}, "effect": {"decision": "require_review", "reason": "drift_review_required", "required_action": "record_drift_review"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/aicr/dashboard", "component": "AICRDashboard", "permission": "aicr:view", "nav_group": "Overview"},
	{"name": "services", "path": "/aicr/services", "component": "AIServiceRegistry", "permission": "aicr:manage_services", "nav_group": "Services"},
	{"name": "providers", "path": "/aicr/providers", "component": "AIProviderRegistry", "permission": "aicr:manage_services", "nav_group": "Services"},
	{"name": "models", "path": "/aicr/models", "component": "ModelCatalog", "permission": "aicr:view_models", "nav_group": "Runtime"},
	{"name": "inference", "path": "/aicr/inference", "component": "InferenceConsole", "permission": "aicr:run_inference", "nav_group": "Runtime"},
	{"name": "workflows", "path": "/aicr/workflows", "component": "AIWorkflowDesigner", "permission": "aicr:manage_workflows", "nav_group": "Orchestration"},
	{"name": "agent_runtimes", "path": "/aicr/agent-runtimes", "component": "AgentRuntimeConsole", "permission": "aicr:manage_agents", "nav_group": "Orchestration"},
	{"name": "agents", "path": "/aicr/agents", "component": "AIAgentRoster", "permission": "aicr:manage_agents", "nav_group": "Orchestration"},
	{"name": "lifecycle", "path": "/aicr/lifecycle", "component": "AICRLifecycleBatchMonitor", "permission": "aicr:govern", "nav_group": "Operations"},
	{"name": "governance", "path": "/aicr/governance", "component": "AIGovernanceCenter", "permission": "aicr:govern", "nav_group": "Governance"},
	{"name": "evaluations", "path": "/aicr/evaluations", "component": "ModelEvaluationConsole", "permission": "aicr:govern", "nav_group": "Governance"},
	{"name": "metrics", "path": "/aicr/metrics", "component": "AICRMetrics", "permission": "aicr:view_metrics", "nav_group": "Operations"},
	{"name": "audit", "path": "/aicr/audit", "component": "AICRAuditTimeline", "permission": "aicr:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/aicr/settings", "component": "AICRSettings", "permission": "aicr:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "aicr_ai_control_console",
	"tokens": {
		"color.primary": "#243B6B",
		"color.accent": "#C86F2D",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F6F7FB",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"ai_service_card": {"icon": "brain-circuit", "status_indicator": "health-pill", "risk_style": "policy-band"},
		"provider_registry_row": {"icon": "server-cog", "status_indicator": "provider-chip", "risk_style": "egress-band"},
		"model_catalog_row": {"icon": "box", "status_indicator": "lifecycle-pill", "risk_style": "evaluation-band"},
		"inference_trace_panel": {"visual": "request-timeline", "highlight": "latency-chip"},
		"workflow_graph": {"visual": "directed-agent-graph", "edge_style": "handoff-line"},
		"agent_runtime_card": {"icon": "bot", "status_indicator": "runtime-chip", "risk_style": "tool-policy-band"},
		"ai_agent_roster": {"icon": "bot-message-square", "status_indicator": "agent-approval-chip", "risk_style": "scope-band"},
		"bytewax_lifecycle_panel": {"icon": "git-branch", "status_indicator": "stream-chip", "risk_style": "processor-band"},
		"governance_rule_stack": {"visual": "rule-ladder", "status_style": "decision-chip"},
		"evaluation_console": {"visual": "score-grid", "highlight": "drift-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "evidence-chip"},
	},
}


def agent_manifest() -> dict[str, Any]:
	"""Return first-class AICR agent composition manifest."""
	return {
		"first_class": True,
		"supported_runtimes": list(SUPPORTED_AICR_AGENT_RUNTIMES),
		"supported_roles": list(SUPPORTED_AICR_AGENT_ROLES),
		"privileged_roles": list(PRIVILEGED_AICR_AGENT_ROLES),
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
		"adapter_contract": "provider_neutral_cli_or_api_adapter",
	}


def streaming_manifest() -> dict[str, Any]:
	"""Return AICR lifecycle stream-processing contract."""
	return {
		"engine": "bytewax",
		"lifecycle_stream": "aicr.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"model_batch",
			"prompt_batch",
			"inference_batch",
			"evaluation_batch",
			"safety_batch",
			"routing_batch",
			"ai_agent_batch",
		],
		"topics": [
			"aicr.models",
			"aicr.prompts",
			"aicr.inference",
			"aicr.evaluations",
			"aicr.safety",
			"aicr.routing",
			"aicr.agents",
		],
		"broker_core_dependency_allowed": False,
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable AICR capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "aicr",
		"display_name": "AI Core Framework",
		"provides": ["ai_core", "model_inference", "ai_agent_composition"],
		"requires": ["conf", "auth", "mqeb", "moni"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/aicr/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"agents": agent_manifest(),
		"streaming": streaming_manifest(),
		"theme": deepcopy(THEME),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default AICR governance rules."""
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
		if key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
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
