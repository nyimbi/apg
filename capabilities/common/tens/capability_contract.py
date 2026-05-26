"""Executable capability contract for APG Tenants Legacy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"legacy_mapping": {"legacy_owner_required": True, "mapping_validation_required": True, "source_system_required": True, "compatibility_scope_required": True},
	"migration": {"migration_plan_required": True, "approval_required": True, "rollback_plan_required": True, "post_migration_validation_required": True},
	"access": {"auth_boundary_required": True, "tenant_isolation_validation_required": True, "legacy_role_mapping_required": True, "privileged_access_review_required": True},
	"governance": {"require_tenant_context": True, "audit_legacy_tenant_changes": True, "stale_tenant_review_days": 180, "deprecation_plan_required": True},
	"ui": {"enable_legacy_tenant_registry": True, "enable_mapping_workbench": True, "enable_migration_queue": True, "enable_boundary_review": True},
	"theme": {"default_theme": "tens_legacy_tenant_migration", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": ["tenant_id", "legacy_mapping", "migration", "access", "governance", "ui", "theme"],
	"properties": {key: {"type": "object"} for key in ["legacy_mapping", "migration", "access", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All legacy tenant operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "legacy_tenant_requires_owner", "description": "Legacy tenants require an accountable owner.", "condition": {"operation": "register_legacy_tenant", "legacy_owner_assigned": False}, "effect": {"decision": "deny", "reason": "legacy_owner_required", "required_action": "assign_legacy_owner"}},
	{"name": "mapping_requires_validation", "description": "Tenant mappings require validation.", "condition": {"operation": "map_tenant", "mapping_validated": False}, "effect": {"decision": "deny", "reason": "mapping_validation_required", "required_action": "validate_tenant_mapping"}},
	{"name": "migration_requires_approval", "description": "Legacy tenant migrations require approval.", "condition": {"operation": "migrate_tenant", "approval_recorded": False}, "effect": {"decision": "deny", "reason": "migration_approval_required", "required_action": "record_migration_approval"}},
	{"name": "access_boundary_required", "description": "Legacy tenant access requires validated auth boundaries.", "condition": {"auth_boundary_validated": False}, "effect": {"decision": "deny", "reason": "auth_boundary_required", "required_action": "validate_auth_boundary"}},
	{"name": "stale_legacy_tenant_requires_review", "description": "Stale legacy tenants require review.", "condition": {"days_since_activity_gt": 180, "stale_review_recorded": False}, "effect": {"decision": "require_review", "reason": "stale_legacy_tenant_review_required", "required_action": "review_legacy_tenant"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/tens/dashboard", "component": "TENSDashboard", "permission": "tens:view", "nav_group": "Overview"},
	{"name": "tenants", "path": "/tens/tenants", "component": "LegacyTenantRegistry", "permission": "tens:view", "nav_group": "Tenants"},
	{"name": "mappings", "path": "/tens/mappings", "component": "TenantMappingWorkbench", "permission": "tens:map", "nav_group": "Mapping"},
	{"name": "migrations", "path": "/tens/migrations", "component": "MigrationQueue", "permission": "tens:migrate", "nav_group": "Migration"},
	{"name": "boundaries", "path": "/tens/boundaries", "component": "BoundaryReview", "permission": "tens:approve", "nav_group": "Access"},
	{"name": "deprecation", "path": "/tens/deprecation", "component": "DeprecationPlan", "permission": "tens:approve", "nav_group": "Governance"},
	{"name": "audit", "path": "/tens/audit", "component": "LegacyTenantAudit", "permission": "tens:view", "nav_group": "Governance"},
	{"name": "settings", "path": "/tens/settings", "component": "TENSSettings", "permission": "tens:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {
	"name": "tens_legacy_tenant_migration",
	"tokens": {"color.primary": "#28536B", "color.accent": "#B7791F", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {"tenant_card": {"icon": "building-2", "status_indicator": "legacy-pill", "risk_style": "migration-band"}, "mapping_workbench": {"visual": "mapping-table", "highlight": "validation-chip"}, "migration_queue": {"visual": "migration-lanes", "status_style": "approval-chip"}, "boundary_review": {"visual": "access-matrix", "status_style": "isolation-chip"}}
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "tens", "display_name": "Tenants Legacy", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "flask_appbuilder", "view_module": "views.py", "api_prefix": "/tens/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
