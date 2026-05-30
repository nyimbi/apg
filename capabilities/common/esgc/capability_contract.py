"""Executable capability contract for APG ESG and Carbon Tracking."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_ESGC_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ESGC_AGENT_ROLES = [
	"inventory_reviewer",
	"factor_reviewer",
	"activity_reviewer",
	"report_reviewer",
	"target_reviewer",
]
SUPPORTED_SCOPES = ["scope_1", "scope_2", "scope_3"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"emissions": {
		"organization_owner_required": True,
		"scope_classification_required": True,
		"supported_scopes": SUPPORTED_SCOPES,
		"activity_data_validation_required": True,
		"geospatial_boundary_required": True,
		"activity_evidence_required": True,
	},
	"data_sources": {
		"approved_factor_source_required": True,
		"source_evidence_required": True,
		"factor_versioning_required": True,
		"forecasting_enabled": True,
		"iot_metering_supported": True,
	},
	"reporting": {
		"report_approval_required": True,
		"compliance_mapping_required": True,
		"audit_evidence_required": True,
		"target_tracking_enabled": True,
	},
	"targets": {
		"baseline_required": True,
		"target_year_required": True,
		"progress_calculation_required": True,
	},
	"esgc_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_role_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_ESGC_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_ESGC_AGENT_ROLES,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_emission_changes": True,
		"anomaly_review_required": True,
		"factor_versioning_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"emissions_metrics_required": True,
		"target_metrics_required": True,
		"agent_activity_required": True,
		"event_stream": "bytewax",
	},
	"adapters": {
		"generated_app_runtime": "service.EsgcService",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"audit_sink": "audl",
		"identity": "auth",
		"configuration": "conf",
		"geospatial": "geos",
		"prediction": "pred",
		"compliance": "comp",
		"metering": "iotd",
	},
	"ui": {
		"enable_emissions_dashboard": True,
		"enable_factor_library": True,
		"enable_data_sources": True,
		"enable_report_builder": True,
		"enable_target_tracker": True,
		"enable_agent_panel": True,
		"enable_rules": True,
		"enable_audit": True,
	},
	"theme": {
		"default_theme": "esgc_sustainability_ops",
		"allow_tenant_overrides": True,
	},
}


CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"emissions",
		"data_sources",
		"reporting",
		"targets",
		"esgc_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		key: {"type": "object"}
		for key in [
			"emissions",
			"data_sources",
			"reporting",
			"targets",
			"esgc_agents",
			"governance",
			"observability",
			"adapters",
			"ui",
			"theme",
		]
	}
	| {"tenant_id": {"type": "string", "minLength": 1}},
}


RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ESG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "inventory_requires_owner", "description": "Emissions inventories require an accountable owner.", "condition": {"operation": "create_inventory", "organization_owner_assigned": False}, "effect": {"decision": "deny", "reason": "organization_owner_required", "required_action": "assign_inventory_owner"}},
	{"name": "inventory_requires_boundary", "description": "Emissions inventories require a reporting boundary.", "condition": {"operation": "create_inventory", "boundary_present": False}, "effect": {"decision": "deny", "reason": "boundary_required", "required_action": "attach_reporting_boundary"}},
	{"name": "factor_requires_approved_source", "description": "Emission factors require approved sources.", "condition": {"factor_source_approved": False}, "effect": {"decision": "deny", "reason": "factor_source_required", "required_action": "attach_approved_factor_source"}},
	{"name": "factor_requires_source_evidence", "description": "Emission factors require source evidence.", "condition": {"operation": "register_factor", "source_evidence_present": False}, "effect": {"decision": "deny", "reason": "source_evidence_required", "required_action": "attach_source_evidence"}},
	{"name": "factor_requires_version", "description": "Emission factors require version metadata.", "condition": {"operation": "register_factor", "factor_version_present": False}, "effect": {"decision": "deny", "reason": "factor_version_required", "required_action": "set_factor_version"}},
	{"name": "emission_requires_boundary", "description": "Emissions records require reporting boundary.", "condition": {"geospatial_boundary_present": False}, "effect": {"decision": "deny", "reason": "boundary_required", "required_action": "attach_reporting_boundary"}},
	{"name": "activity_requires_evidence", "description": "Emission activity requires evidence reference.", "condition": {"operation": "record_activity", "activity_evidence_present": False}, "effect": {"decision": "deny", "reason": "activity_evidence_required", "required_action": "attach_activity_evidence"}},
	{"name": "report_requires_approval", "description": "ESG reports require approval.", "condition": {"operation": "publish_report", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "record_report_approval"}},
	{"name": "report_requires_compliance_mapping", "description": "ESG reports require compliance mapping.", "condition": {"operation": "publish_report", "compliance_mapping_present": False}, "effect": {"decision": "deny", "reason": "compliance_mapping_required", "required_action": "attach_compliance_mapping"}},
	{"name": "report_requires_audit_evidence", "description": "ESG reports require audit evidence.", "condition": {"operation": "publish_report", "audit_evidence_present": False}, "effect": {"decision": "deny", "reason": "audit_evidence_required", "required_action": "attach_audit_evidence"}},
	{"name": "target_requires_baseline", "description": "Reduction targets require a baseline.", "condition": {"operation": "create_target", "baseline_present": False}, "effect": {"decision": "deny", "reason": "target_baseline_required", "required_action": "attach_target_baseline"}},
	{"name": "emission_anomaly_requires_review", "description": "Emission anomalies require review.", "condition": {"emission_anomaly_detected": True, "anomaly_review_recorded": False}, "effect": {"decision": "require_review", "reason": "emission_anomaly_review_required", "required_action": "review_emission_anomaly"}},
	{"name": "esgc_agent_requires_registration", "description": "AI ESG agents must be registered.", "condition": {"esgc_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "esgc_agent_registration_required", "required_action": "register_esgc_agent"}},
	{"name": "esgc_agent_runtime_supported", "description": "AI ESG agents must use a supported runtime.", "condition": {"esgc_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "esgc_agent_runtime_not_supported", "required_action": "choose_supported_esgc_agent_runtime"}},
	{"name": "esgc_agent_role_supported", "description": "AI ESG agents must use a supported role.", "condition": {"esgc_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "esgc_agent_role_not_supported", "required_action": "choose_supported_esgc_agent_role"}},
	{"name": "esgc_agent_requires_scope", "description": "AI ESG agents require explicit scope.", "condition": {"esgc_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "esgc_agent_scope_required", "required_action": "set_esgc_agent_scope"}},
	{"name": "esgc_agent_requires_disclosure", "description": "AI ESG-agent contributions require disclosure.", "condition": {"esgc_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "esgc_agent_disclosure_required", "required_action": "disclose_esgc_agent"}},
	{"name": "esgc_state_change_requires_audit", "description": "ESG lifecycle state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "esgc_audit_event_required", "required_action": "record_esgc_audit_event"}},
	{"name": "batch_esgc_mutation_requires_bytewax", "description": "Batch ESG mutations must use Bytewax event streams.", "condition": {"requested_operation": "batch_esgc_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]


UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/esgc/dashboard", "component": "EsgcDashboard", "permission": "esgc:view", "nav_group": "Overview"},
	{"name": "emissions", "path": "/esgc/emissions", "component": "EmissionsInventory", "permission": "esgc:manage_data", "nav_group": "Inventory"},
	{"name": "factors", "path": "/esgc/factors", "component": "FactorLibrary", "permission": "esgc:manage_data", "nav_group": "Inventory"},
	{"name": "data_sources", "path": "/esgc/data-sources", "component": "EsgDataSources", "permission": "esgc:manage_data", "nav_group": "Data"},
	{"name": "reports", "path": "/esgc/reports", "component": "ReportBuilder", "permission": "esgc:report", "nav_group": "Reporting"},
	{"name": "targets", "path": "/esgc/targets", "component": "TargetTracker", "permission": "esgc:view", "nav_group": "Targets"},
	{"name": "agents", "path": "/esgc/agents", "component": "EsgcAgentPanel", "permission": "esgc:govern", "nav_group": "Governance"},
	{"name": "rules", "path": "/esgc/rules", "component": "EsgcRules", "permission": "esgc:govern", "nav_group": "Governance"},
	{"name": "audit", "path": "/esgc/audit", "component": "EsgAuditEvidence", "permission": "esgc:approve", "nav_group": "Governance"},
	{"name": "settings", "path": "/esgc/settings", "component": "EsgcSettings", "permission": "esgc:admin", "nav_group": "Administration"},
]


THEME: dict[str, Any] = {
	"name": "esgc_sustainability_ops",
	"tokens": {
		"color.primary": "#22543D",
		"color.accent": "#2C5282",
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
		"emissions_card": {"icon": "leaf", "status_indicator": "scope-pill", "risk_style": "carbon-band"},
		"factor_library": {"visual": "factor-table", "highlight": "source-chip"},
		"report_builder": {"visual": "evidence-checklist", "status_style": "approval-chip"},
		"target_tracker": {"visual": "reduction-chart", "status_style": "forecast-chip"},
		"esgc_agent_panel": {"icon": "bot", "status_indicator": "scope-chip"},
		"stream_health": {"visual": "event-lane", "status_style": "stream-chip"},
		"audit": {"visual": "event-ledger", "status_style": "evidence-chip"},
	},
}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"topic": "apg.esgc.lifecycle",
		"state": ["inventories", "factors", "activities", "reports", "targets", "esgc_agents", "audit_events"],
		"events": [
			"esgc_inventory_created",
			"esgc_factor_registered",
			"esgc_activity_recorded",
			"esgc_report_published",
			"esgc_target_created",
			"esgc_agent_registered",
		],
		"batch_mutation_guardrail": "batch_esgc_mutation_requires_bytewax",
	}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "esgc",
		"display_name": "ESG and Carbon Tracking",
		"version": "1.0.0",
		"provides": [
			"emissions_inventory",
			"factor_library",
			"activity_emissions",
			"sustainability_reporting",
			"target_tracking",
			"esg_evidence",
			"esgc_agents",
		],
		"requires": ["auth", "conf", "audl", "geos", "pred", "comp"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/esgc/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


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
