"""Executable capability contract for APG Scraper/Data Harvesting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {"source_owner_required": True, "terms_evidence_required": True, "credential_vault_required": True, "rate_limit_required": True},
	"extraction": {"schema_validation_required": True, "pipeline_handoff_required": True, "incremental_mode_supported": True, "result_retention_days": 30},
	"compliance": {"robots_policy_required": True, "pii_handling_policy_required": True, "restricted_source_review_required": True, "audit_harvest_runs": True},
	"governance": {"require_tenant_context": True, "approval_for_sensitive_sources": True, "job_schedule_policy_required": True, "dlp_scan_required": True},
	"ui": {"enable_source_console": True, "enable_job_monitor": True, "enable_extractor_workbench": True, "enable_compliance_review": True},
	"theme": {"default_theme": "scrp_harvest_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "sources", "extraction", "compliance", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["sources", "extraction", "compliance", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All harvesting operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "source_requires_owner", "description": "Harvest sources require an accountable owner.", "condition": {"operation": "register_source", "source_owner_assigned": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_terms_required", "description": "Harvest sources require terms or authorization evidence.", "condition": {"terms_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_terms_required", "required_action": "attach_terms_evidence"}},
	{"name": "pii_requires_handling_policy", "description": "PII-bearing harvests require a handling policy.", "condition": {"pii_expected": True, "pii_policy_attached": False}, "effect": {"decision": "deny", "reason": "pii_policy_required", "required_action": "attach_pii_policy"}},
	{"name": "harvest_requires_schedule_policy", "description": "Harvest jobs require rate and schedule policy.", "condition": {"operation": "run_harvest", "schedule_policy_attached": False}, "effect": {"decision": "deny", "reason": "schedule_policy_required", "required_action": "attach_schedule_policy"}},
	{"name": "sensitive_source_requires_review", "description": "Sensitive sources require review.", "condition": {"sensitive_source": True, "source_review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_source_review_required", "required_action": "review_source"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/scrp/dashboard", "component": "SCRPDashboard", "permission": "scrp:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/scrp/sources", "component": "SourceRegistry", "permission": "scrp:configure_sources", "nav_group": "Sources"},
	{"name": "jobs", "path": "/scrp/jobs", "component": "HarvestJobMonitor", "permission": "scrp:run_jobs", "nav_group": "Jobs"},
	{"name": "extractors", "path": "/scrp/extractors", "component": "ExtractorWorkbench", "permission": "scrp:configure_sources", "nav_group": "Extraction"},
	{"name": "pipelines", "path": "/scrp/pipelines", "component": "PipelineHandoff", "permission": "scrp:view", "nav_group": "Extraction"},
	{"name": "compliance", "path": "/scrp/compliance", "component": "HarvestCompliance", "permission": "scrp:approve_harvests", "nav_group": "Governance"},
	{"name": "results", "path": "/scrp/results", "component": "HarvestResults", "permission": "scrp:view", "nav_group": "Results"},
	{"name": "settings", "path": "/scrp/settings", "component": "SCRPSettings", "permission": "scrp:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "scrp_harvest_ops",
	"tokens": {"color.primary": "#28536B", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"source_card": {"icon": "database-zap", "status_indicator": "source-pill", "risk_style": "terms-band"}, "job_monitor": {"visual": "harvest-timeline", "highlight": "rate-chip"}, "extractor_workbench": {"visual": "schema-mapper", "status_style": "parser-chip"}, "compliance_panel": {"visual": "policy-checklist", "status_style": "approval-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "scrp", "display_name": "Scraper/Data Harvesting", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/scrp/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
