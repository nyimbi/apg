"""Executable capability contract for APG Continuous Integration/Delivery."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"pipelines": {"pipeline_owner_required": True, "versioning_enabled": True, "source_policy_required": True, "max_parallel_jobs": 100},
	"builds": {"worker_pool_required": True, "secret_scope_required": True, "log_trace_capture_required": True, "cache_policy_required": True},
	"gates": {"quality_gate_required": True, "security_scan_required": True, "artifact_signature_required": True, "promotion_approval_required": True},
	"governance": {"require_tenant_context": True, "audit_pipeline_runs": True, "environment_policy_required": True, "separation_of_duties_required": True},
	"ui": {"enable_pipeline_console": True, "enable_build_monitor": True, "enable_artifact_registry": True, "enable_gate_dashboard": True},
	"theme": {"default_theme": "cicd_pipeline_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "pipelines", "builds", "gates", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["pipelines", "builds", "gates", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All CI/CD operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "pipeline_requires_owner", "description": "Pipelines require an accountable owner.", "condition": {"operation": "create_pipeline", "pipeline_owner_assigned": False}, "effect": {"decision": "deny", "reason": "pipeline_owner_required", "required_action": "assign_pipeline_owner"}},
	{"name": "build_requires_secret_scope", "description": "Builds require secret scope policy.", "condition": {"operation": "run_build", "secret_scope_attached": False}, "effect": {"decision": "deny", "reason": "secret_scope_required", "required_action": "attach_secret_scope"}},
	{"name": "artifact_requires_signature", "description": "Promotion artifacts require signatures.", "condition": {"artifact_promotion_requested": True, "artifact_signed": False}, "effect": {"decision": "deny", "reason": "artifact_signature_required", "required_action": "sign_artifact"}},
	{"name": "promotion_requires_quality_gate", "description": "Promotions require passing quality gates.", "condition": {"operation": "promote_artifact", "quality_gate_passed": False}, "effect": {"decision": "deny", "reason": "quality_gate_required", "required_action": "pass_quality_gate"}},
	{"name": "high_parallelism_requires_review", "description": "High parallelism requires capacity review.", "condition": {"parallel_job_count_gt": 100, "capacity_review_recorded": False}, "effect": {"decision": "require_review", "reason": "capacity_review_required", "required_action": "review_pipeline_capacity"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/cicd/dashboard", "component": "CICDDashboard", "permission": "cicd:view", "nav_group": "Overview"},
	{"name": "pipelines", "path": "/cicd/pipelines", "component": "PipelineConsole", "permission": "cicd:manage_pipelines", "nav_group": "Pipelines"},
	{"name": "builds", "path": "/cicd/builds", "component": "BuildMonitor", "permission": "cicd:run_builds", "nav_group": "Builds"},
	{"name": "artifacts", "path": "/cicd/artifacts", "component": "ArtifactRegistry", "permission": "cicd:view", "nav_group": "Artifacts"},
	{"name": "gates", "path": "/cicd/gates", "component": "QualityGates", "permission": "cicd:promote", "nav_group": "Release"},
	{"name": "promotions", "path": "/cicd/promotions", "component": "PromotionConsole", "permission": "cicd:promote", "nav_group": "Release"},
	{"name": "analytics", "path": "/cicd/analytics", "component": "PipelineAnalytics", "permission": "cicd:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/cicd/settings", "component": "CICDSettings", "permission": "cicd:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "cicd_pipeline_ops", "tokens": {"color.primary": "#2C5282", "color.accent": "#DD6B20", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"pipeline_graph": {"icon": "git-branch", "status_indicator": "pipeline-pill", "risk_style": "gate-band"}, "build_monitor": {"visual": "build-list", "highlight": "trace-chip"}, "artifact_registry": {"visual": "artifact-table", "status_style": "signature-chip"}, "quality_gate": {"visual": "gate-checklist", "status_style": "scan-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "cicd", "display_name": "Continuous Integration and Delivery", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/cicd/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
