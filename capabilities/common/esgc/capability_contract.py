"""Executable capability contract for APG ESG/Carbon Tracking."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"emissions": {"organization_owner_required": True, "scope_classification_required": True, "activity_data_validation_required": True, "geospatial_boundary_required": True},
	"data_sources": {"approved_factor_source_required": True, "source_evidence_required": True, "forecasting_enabled": True, "iot_metering_supported": True},
	"reporting": {"report_approval_required": True, "compliance_mapping_required": True, "audit_evidence_required": True, "target_tracking_enabled": True},
	"governance": {"require_tenant_context": True, "audit_emission_changes": True, "anomaly_review_required": True, "factor_versioning_required": True},
	"ui": {"enable_emissions_dashboard": True, "enable_factor_library": True, "enable_report_builder": True, "enable_target_tracker": True},
	"theme": {"default_theme": "esgc_sustainability_ops", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "emissions", "data_sources", "reporting", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["emissions", "data_sources", "reporting", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All ESG operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "inventory_requires_owner", "description": "Emissions inventories require an accountable owner.", "condition": {"operation": "create_inventory", "organization_owner_assigned": False}, "effect": {"decision": "deny", "reason": "organization_owner_required", "required_action": "assign_inventory_owner"}},
	{"name": "factor_requires_approved_source", "description": "Emission factors require approved sources.", "condition": {"factor_source_approved": False}, "effect": {"decision": "deny", "reason": "factor_source_required", "required_action": "attach_approved_factor_source"}},
	{"name": "emission_requires_boundary", "description": "Emissions records require reporting boundary.", "condition": {"geospatial_boundary_present": False}, "effect": {"decision": "deny", "reason": "boundary_required", "required_action": "attach_reporting_boundary"}},
	{"name": "report_requires_approval", "description": "ESG reports require approval.", "condition": {"operation": "publish_report", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "report_approval_required", "required_action": "record_report_approval"}},
	{"name": "emission_anomaly_requires_review", "description": "Emission anomalies require review.", "condition": {"emission_anomaly_detected": True, "anomaly_review_recorded": False}, "effect": {"decision": "require_review", "reason": "emission_anomaly_review_required", "required_action": "review_emission_anomaly"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/esgc/dashboard", "component": "ESGCDashboard", "permission": "esgc:view", "nav_group": "Overview"},
	{"name": "emissions", "path": "/esgc/emissions", "component": "EmissionsInventory", "permission": "esgc:manage_data", "nav_group": "Inventory"},
	{"name": "factors", "path": "/esgc/factors", "component": "FactorLibrary", "permission": "esgc:manage_data", "nav_group": "Inventory"},
	{"name": "data_sources", "path": "/esgc/data-sources", "component": "ESGDataSources", "permission": "esgc:manage_data", "nav_group": "Data"},
	{"name": "reports", "path": "/esgc/reports", "component": "ReportBuilder", "permission": "esgc:report", "nav_group": "Reporting"},
	{"name": "targets", "path": "/esgc/targets", "component": "TargetTracker", "permission": "esgc:view", "nav_group": "Targets"},
	{"name": "audit", "path": "/esgc/audit", "component": "ESGAuditEvidence", "permission": "esgc:approve", "nav_group": "Governance"},
	{"name": "settings", "path": "/esgc/settings", "component": "ESGCSettings", "permission": "esgc:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "esgc_sustainability_ops",
	"tokens": {"color.primary": "#22543D", "color.accent": "#2C5282", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"emissions_card": {"icon": "leaf", "status_indicator": "scope-pill", "risk_style": "carbon-band"}, "factor_library": {"visual": "factor-table", "highlight": "source-chip"}, "report_builder": {"visual": "evidence-checklist", "status_style": "approval-chip"}, "target_tracker": {"visual": "reduction-chart", "status_style": "forecast-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "esgc", "display_name": "ESG/Carbon Tracking", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/esgc/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
