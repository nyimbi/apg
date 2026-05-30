"""Executable capability contract for APG Scraper/Data Harvesting."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_HARVEST_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_HARVEST_AGENT_ROLES = ["source_reviewer", "extractor_designer", "compliance_reviewer", "run_operator", "pipeline_operator"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"sources": {"source_owner_required": True, "terms_evidence_required": True, "credential_vault_required": True, "rate_limit_required": True},
	"extraction": {"schema_validation_required": True, "pipeline_handoff_required": True, "incremental_mode_supported": True, "result_retention_days": 30},
	"compliance": {"robots_policy_required": True, "pii_handling_policy_required": True, "restricted_source_review_required": True, "audit_harvest_runs": True},
	"harvest_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_HARVEST_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_HARVEST_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "approval_for_sensitive_sources": True, "job_schedule_policy_required": True, "dlp_scan_required": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "quality_metrics_required": True, "latency_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.ScrpService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "connections": "conn", "etl_pipeline": "etlp", "auth": "auth", "scheduler": "schd", "dlp": "dlpd", "audit_sink": "audl"},
	"ui": {"enable_source_console": True, "enable_job_monitor": True, "enable_extractor_workbench": True, "enable_compliance_review": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "scrp_harvest_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "sources", "extraction", "compliance", "harvest_agents", "governance", "observability", "adapters", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["sources", "extraction", "compliance", "harvest_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All harvesting operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "source_requires_owner", "description": "Harvest sources require an accountable owner.", "condition": {"operation": "register_source", "source_owner_assigned": False}, "effect": {"decision": "deny", "reason": "source_owner_required", "required_action": "assign_source_owner"}},
	{"name": "source_terms_required", "description": "Harvest sources require terms or authorization evidence.", "condition": {"terms_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_terms_required", "required_action": "attach_terms_evidence"}},
	{"name": "pii_requires_handling_policy", "description": "PII-bearing harvests require a handling policy.", "condition": {"pii_expected": True, "pii_policy_attached": False}, "effect": {"decision": "deny", "reason": "pii_policy_required", "required_action": "attach_pii_policy"}},
	{"name": "harvest_requires_schedule_policy", "description": "Harvest jobs require rate and schedule policy.", "condition": {"operation": "run_harvest", "schedule_policy_attached": False}, "effect": {"decision": "deny", "reason": "schedule_policy_required", "required_action": "attach_schedule_policy"}},
	{"name": "sensitive_source_requires_review", "description": "Sensitive sources require review.", "condition": {"sensitive_source": True, "source_review_recorded": False}, "effect": {"decision": "require_review", "reason": "sensitive_source_review_required", "required_action": "review_source"}},
	{"name": "credential_vault_required", "description": "Harvest sources require credential vault references.", "condition": {"operation": "register_source", "credential_vault_present": False}, "effect": {"decision": "deny", "reason": "credential_vault_required", "required_action": "attach_credential_vault_ref"}},
	{"name": "robots_policy_required", "description": "Web and feed sources require robots/terms policy evidence.", "condition": {"robots_policy_attached": False}, "effect": {"decision": "deny", "reason": "robots_policy_required", "required_action": "attach_robots_policy"}},
	{"name": "rate_limit_required", "description": "Harvest sources require positive rate limits.", "condition": {"operation": "register_source", "rate_limit_per_minute_lte": 0}, "effect": {"decision": "deny", "reason": "rate_limit_required", "required_action": "set_rate_limit"}},
	{"name": "extractor_schema_required", "description": "Extractor profiles require schema validation.", "condition": {"operation": "create_extractor", "schema_present": False}, "effect": {"decision": "deny", "reason": "schema_validation_required", "required_action": "attach_extractor_schema"}},
	{"name": "pipeline_handoff_required", "description": "Harvest jobs require downstream pipeline handoff.", "condition": {"operation": "create_harvest_job", "pipeline_target_present": False}, "effect": {"decision": "deny", "reason": "pipeline_handoff_required", "required_action": "set_pipeline_target"}},
	{"name": "dlp_scan_required", "description": "PII-bearing harvest runs require DLP scans before completion.", "condition": {"operation": "complete_harvest_run", "pii_expected": True, "dlp_scanned": False}, "effect": {"decision": "deny", "reason": "dlp_scan_required", "required_action": "run_dlp_scan"}},
	{"name": "harvest_agent_requires_registration", "description": "AI harvest agents must be registered.", "condition": {"harvest_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "harvest_agent_registration_required", "required_action": "register_harvest_agent"}},
	{"name": "harvest_agent_runtime_supported", "description": "AI harvest agents must use a supported runtime.", "condition": {"harvest_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "harvest_agent_runtime_not_supported", "required_action": "choose_supported_harvest_agent_runtime"}},
	{"name": "harvest_agent_role_supported", "description": "AI harvest agents must use a supported role.", "condition": {"harvest_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "harvest_agent_role_not_supported", "required_action": "choose_supported_harvest_agent_role"}},
	{"name": "harvest_agent_requires_scope", "description": "AI harvest agents require explicit scope.", "condition": {"harvest_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "harvest_agent_scope_required", "required_action": "set_harvest_agent_scope"}},
	{"name": "harvest_agent_requires_disclosure", "description": "AI harvest-agent contributions require disclosure.", "condition": {"harvest_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "harvest_agent_disclosure_required", "required_action": "disclose_harvest_agent"}},
	{"name": "scrp_state_change_requires_reason", "description": "Harvest lifecycle state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "scrp_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "scrp_state_change_requires_audit", "description": "Harvest lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "scrp_audit_event_required", "required_action": "record_scrp_audit_event"}},
	{"name": "cross_tenant_harvest_access_denied", "description": "Harvest records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_harvest_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_harvest_mutation_requires_bytewax", "description": "Batch harvest mutations must use Bytewax event streams.", "condition": {"operation": "batch_harvest_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/scrp/dashboard", "component": "SCRPDashboard", "permission": "scrp:view", "nav_group": "Overview"},
	{"name": "sources", "path": "/scrp/sources", "component": "SourceRegistry", "permission": "scrp:configure_sources", "nav_group": "Sources"},
	{"name": "jobs", "path": "/scrp/jobs", "component": "HarvestJobMonitor", "permission": "scrp:run_jobs", "nav_group": "Jobs"},
	{"name": "extractors", "path": "/scrp/extractors", "component": "ExtractorWorkbench", "permission": "scrp:configure_sources", "nav_group": "Extraction"},
	{"name": "pipelines", "path": "/scrp/pipelines", "component": "PipelineHandoff", "permission": "scrp:view", "nav_group": "Extraction"},
	{"name": "compliance", "path": "/scrp/compliance", "component": "HarvestCompliance", "permission": "scrp:approve_harvests", "nav_group": "Governance"},
	{"name": "results", "path": "/scrp/results", "component": "HarvestResults", "permission": "scrp:view", "nav_group": "Results"},
	{"name": "agents", "path": "/scrp/agents", "component": "HarvestAgentPanel", "permission": "scrp:approve_harvests", "nav_group": "Agents"},
	{"name": "audit", "path": "/scrp/audit", "component": "HarvestAuditTrail", "permission": "scrp:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/scrp/analytics", "component": "HarvestAnalytics", "permission": "scrp:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/scrp/settings", "component": "SCRPSettings", "permission": "scrp:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "scrp_harvest_ops",
	"tokens": {"color.primary": "#28536B", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"source_card": {"icon": "database-zap", "status_indicator": "source-pill", "risk_style": "terms-band"}, "job_monitor": {"visual": "harvest-timeline", "highlight": "rate-chip"}, "extractor_workbench": {"visual": "schema-mapper", "status_style": "parser-chip"}, "compliance_panel": {"visual": "policy-checklist", "status_style": "approval-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-todo", "status_style": "governance-chip"}}
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.scrp.lifecycle",
	"state": ["sources", "extractors", "jobs", "runs", "results", "handoffs", "harvest_agents", "audit_events"],
	"events": ["source_registered", "extractor_created", "harvest_job_created", "harvest_run_started", "harvest_run_completed", "harvest_agent_registered", "harvest_job_state_changed"],
	"batch_mutation_guardrail": "batch_harvest_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "scrp", "display_name": "Scraper/Data Harvesting", "provides": ["source_registry", "harvest_jobs", "extractor_profiles", "compliance_controls", "pipeline_handoff", "harvest_agents"], "requires": ["conn", "etlp", "auth"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/scrp/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
