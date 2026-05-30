"""Executable capability contract for APG Continuous Integration/Delivery."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_DELIVERY_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_DELIVERY_AGENT_ROLES = ["pipeline_designer", "build_operator", "security_reviewer", "release_manager", "incident_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"pipelines": {"pipeline_owner_required": True, "versioning_enabled": True, "source_policy_required": True, "max_parallel_jobs": 100},
	"builds": {"worker_pool_required": True, "secret_scope_required": True, "log_trace_capture_required": True, "cache_policy_required": True},
	"gates": {"quality_gate_required": True, "security_scan_required": True, "artifact_signature_required": True, "promotion_approval_required": True},
	"delivery_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_DELIVERY_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_DELIVERY_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "audit_pipeline_runs": True, "environment_policy_required": True, "separation_of_duties_required": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "quality_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.CicdService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "deployment": "depl", "environment": "envm", "logs": "logt", "secrets": "scpt", "notifications": "ntfy", "audit_sink": "audl", "monitoring": "moni"},
	"ui": {"enable_pipeline_console": True, "enable_build_monitor": True, "enable_artifact_registry": True, "enable_gate_dashboard": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "cicd_pipeline_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "pipelines", "builds", "gates", "delivery_agents", "governance", "observability", "adapters", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["pipelines", "builds", "gates", "delivery_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All CI/CD operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pipeline_requires_owner", "description": "Pipelines require an accountable owner.", "condition": {"operation": "create_pipeline", "pipeline_owner_assigned": False}, "effect": {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}},
	{"name": "build_requires_secret_scope", "description": "Builds require secret scope policy.", "condition": {"operation": "run_build", "secret_scope_attached": False}, "effect": {"decision": "deny", "reason": "secret_scope_required", "required_action": "attach_secret_scope"}},
	{"name": "artifact_requires_signature", "description": "Promotion artifacts require signatures.", "condition": {"artifact_promotion_requested": True, "artifact_signed": False}, "effect": {"decision": "deny", "reason": "artifact_signature_required", "required_action": "sign_artifact"}},
	{"name": "promotion_requires_quality_gate", "description": "Promotions require passing quality gates.", "condition": {"operation": "promote_artifact", "quality_gate_passed": False}, "effect": {"decision": "deny", "reason": "quality_gate_required", "required_action": "pass_quality_gate"}},
	{"name": "high_parallelism_requires_review", "description": "High parallelism requires capacity review.", "condition": {"parallel_job_count_gt": 100, "capacity_review_recorded": False}, "effect": {"decision": "require_review", "reason": "capacity_review_required", "required_action": "review_pipeline_capacity"}},
	{"name": "pipeline_requires_source_policy", "description": "Pipelines require source policy references.", "condition": {"operation": "create_pipeline", "source_policy_attached": False}, "effect": {"decision": "deny", "reason": "source_policy_required", "required_action": "attach_source_policy"}},
	{"name": "pipeline_requires_worker_pool", "description": "Pipelines require worker pool assignment.", "condition": {"operation": "create_pipeline", "worker_pool_attached": False}, "effect": {"decision": "deny", "reason": "worker_pool_required", "required_action": "assign_worker_pool"}},
	{"name": "pipeline_requires_stages", "description": "Pipelines require at least one stage.", "condition": {"operation": "create_pipeline", "stage_count_lte": 0}, "effect": {"decision": "deny", "reason": "pipeline_stages_required", "required_action": "define_pipeline_stages"}},
	{"name": "pipeline_requires_secret_scope", "description": "Pipelines require secret scope policy.", "condition": {"operation": "create_pipeline", "secret_scope_attached": False}, "effect": {"decision": "deny", "reason": "secret_scope_required", "required_action": "attach_secret_scope"}},
	{"name": "pipeline_requires_cache_policy", "description": "Pipelines require build cache policy.", "condition": {"operation": "create_pipeline", "cache_policy_attached": False}, "effect": {"decision": "deny", "reason": "cache_policy_required", "required_action": "attach_cache_policy"}},
	{"name": "pipeline_requires_quality_gate", "description": "Pipelines require quality gate policy.", "condition": {"operation": "create_pipeline", "quality_gate_attached": False}, "effect": {"decision": "deny", "reason": "quality_gate_required", "required_action": "attach_quality_gate"}},
	{"name": "build_requires_trace_capture", "description": "Builds require trace and log capture.", "condition": {"operation": "run_build", "log_trace_captured": False}, "effect": {"decision": "deny", "reason": "log_trace_capture_required", "required_action": "enable_log_trace_capture"}},
	{"name": "gate_requires_security_scan", "description": "Quality gates require security scan evidence.", "condition": {"operation": "record_quality_gate", "security_scan_passed": False}, "effect": {"decision": "deny", "reason": "security_scan_required", "required_action": "run_security_scan"}},
	{"name": "promotion_requires_approval", "description": "Promotions require explicit approval evidence.", "condition": {"operation": "promote_artifact", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "promotion_approval_required", "required_action": "record_promotion_approval"}},
	{"name": "promotion_requires_environment_policy", "description": "Promotions require source and target environment policy.", "condition": {"operation": "promote_artifact", "environment_policy_attached": False}, "effect": {"decision": "deny", "reason": "environment_policy_required", "required_action": "attach_environment_policy"}},
	{"name": "promotion_requires_separation_of_duties", "description": "Promotions require release requester and approver separation.", "condition": {"operation": "promote_artifact", "separation_of_duties_met": False}, "effect": {"decision": "deny", "reason": "separation_of_duties_required", "required_action": "assign_independent_approver"}},
	{"name": "delivery_agent_requires_registration", "description": "AI delivery agents must be registered.", "condition": {"delivery_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "delivery_agent_registration_required", "required_action": "register_delivery_agent"}},
	{"name": "delivery_agent_runtime_supported", "description": "AI delivery agents must use a supported runtime.", "condition": {"delivery_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "delivery_agent_runtime_not_supported", "required_action": "choose_supported_delivery_agent_runtime"}},
	{"name": "delivery_agent_role_supported", "description": "AI delivery agents must use a supported role.", "condition": {"delivery_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "delivery_agent_role_not_supported", "required_action": "choose_supported_delivery_agent_role"}},
	{"name": "delivery_agent_requires_scope", "description": "AI delivery agents require explicit scope.", "condition": {"delivery_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "delivery_agent_scope_required", "required_action": "set_delivery_agent_scope"}},
	{"name": "delivery_agent_requires_disclosure", "description": "AI delivery-agent contributions require disclosure.", "condition": {"delivery_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "delivery_agent_disclosure_required", "required_action": "disclose_delivery_agent"}},
	{"name": "cicd_state_change_requires_reason", "description": "Pipeline lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "cicd_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "cicd_state_change_requires_audit", "description": "Pipeline lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "cicd_audit_event_required", "required_action": "record_cicd_audit_event"}},
	{"name": "cross_tenant_pipeline_access_denied", "description": "CI/CD records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_pipeline_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_pipeline_mutation_requires_bytewax", "description": "Batch pipeline mutations must use Bytewax event streams.", "condition": {"operation": "batch_pipeline_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/cicd/dashboard", "component": "CICDDashboard", "permission": "cicd:view", "nav_group": "Overview"},
	{"name": "pipelines", "path": "/cicd/pipelines", "component": "PipelineConsole", "permission": "cicd:manage_pipelines", "nav_group": "Pipelines"},
	{"name": "builds", "path": "/cicd/builds", "component": "BuildMonitor", "permission": "cicd:run_builds", "nav_group": "Builds"},
	{"name": "artifacts", "path": "/cicd/artifacts", "component": "ArtifactRegistry", "permission": "cicd:view", "nav_group": "Artifacts"},
	{"name": "gates", "path": "/cicd/gates", "component": "QualityGates", "permission": "cicd:promote", "nav_group": "Release"},
	{"name": "promotions", "path": "/cicd/promotions", "component": "PromotionConsole", "permission": "cicd:promote", "nav_group": "Release"},
	{"name": "agents", "path": "/cicd/agents", "component": "DeliveryAgentPanel", "permission": "cicd:promote", "nav_group": "Agents"},
	{"name": "audit", "path": "/cicd/audit", "component": "DeliveryAuditTrail", "permission": "cicd:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/cicd/analytics", "component": "PipelineAnalytics", "permission": "cicd:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/cicd/settings", "component": "CICDSettings", "permission": "cicd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "cicd_pipeline_ops", "tokens": {"color.primary": "#2C5282", "color.accent": "#DD6B20", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"pipeline_graph": {"icon": "git-branch", "status_indicator": "pipeline-pill", "risk_style": "gate-band"}, "build_monitor": {"visual": "build-list", "highlight": "trace-chip"}, "artifact_registry": {"visual": "artifact-table", "status_style": "signature-chip"}, "quality_gate": {"visual": "gate-checklist", "status_style": "scan-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.cicd.lifecycle",
	"state": ["pipelines", "builds", "artifacts", "gates", "promotions", "delivery_agents", "audit_events"],
	"events": ["pipeline_created", "pipeline_review_approved", "pipeline_state_changed", "build_run_completed", "artifact_published", "quality_gate_recorded", "artifact_promoted", "delivery_agent_registered"],
	"batch_mutation_guardrail": "batch_pipeline_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "cicd", "display_name": "Continuous Integration and Delivery", "provides": ["pipeline_management", "build_orchestration", "quality_gates", "artifact_promotion", "release_automation", "delivery_agents"], "requires": ["depl", "envm", "logt"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/cicd/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
