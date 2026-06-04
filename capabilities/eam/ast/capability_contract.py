"""Executable capability contract for APG enterprise asset management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_EAM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_EAM_AGENT_ROLES = [
	"asset_reliability_reviewer",
	"maintenance_planner",
	"inspection_reviewer",
	"safety_reviewer",
	"inventory_reviewer",
	"lifecycle_cost_reviewer",
]
EAM_EVENT_STREAM = "apg.eam.ast.lifecycle"


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"assets": {
		"owner_required": True,
		"category_required": True,
		"location_required": True,
		"criticality_required": True,
		"fixed_asset_reference_required_when_capitalized": True,
		"health_score_bounds": [0, 100],
	},
	"locations": {"location_type_required": True, "parent_validation_required": True},
	"maintenance_plans": {
		"strategy_required": True,
		"interval_required": True,
		"condition_source_required_for_predictive": True,
	},
	"work_orders": {
		"asset_required": True,
		"priority_required": True,
		"safety_plan_required": True,
		"approval_required_for_critical": True,
		"completion_outcome_required": True,
	},
	"inspections": {"asset_required": True, "result_required": True, "condition_alert_review_required": True},
	"inventory": {"part_required": True, "positive_quantity_required": True},
	"analytics": {"asset_reliability_enabled": True, "condition_health_scoring_enabled": True},
	"eam_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_EAM_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_EAM_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"safety_review_for_critical_work": True,
	},
	"observability": {
		"event_stream": EAM_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_asset_events": True,
		"emit_location_events": True,
		"emit_maintenance_events": True,
		"emit_work_order_events": True,
		"emit_inspection_events": True,
		"emit_inventory_events": True,
		"emit_condition_events": True,
		"emit_agent_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"fixed_assets": "adapter",
		"theme": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_assets": True,
		"enable_locations": True,
		"enable_maintenance_plans": True,
		"enable_work_orders": True,
		"enable_inspections": True,
		"enable_inventory": True,
		"enable_analytics": True,
		"enable_agents": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "eam_ast_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"assets",
		"locations",
		"maintenance_plans",
		"work_orders",
		"inspections",
		"inventory",
		"analytics",
		"eam_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"assets": {"type": "object"},
		"locations": {"type": "object"},
		"maintenance_plans": {"type": "object"},
		"work_orders": {"type": "object"},
		"inspections": {"type": "object"},
		"inventory": {"type": "object"},
		"analytics": {"type": "object"},
		"eam_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "Asset operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "eam_write_requires_policy", "description": "Asset writes require policy attachment.", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"}},
	{"name": "location_requires_type", "description": "Locations require a location type.", "condition": {"operation": "register_location", "location_type_present": False}, "effect": {"decision": "deny", "reason": "location_type_required", "required_action": "set_location_type"}},
	{"name": "asset_requires_owner", "description": "Assets require an accountable owner.", "condition": {"operation": "register_asset", "asset_owner_assigned": False}, "effect": {"decision": "deny", "reason": "asset_owner_required", "required_action": "assign_asset_owner"}},
	{"name": "asset_requires_category", "description": "Assets require a lifecycle category.", "condition": {"operation": "register_asset", "asset_category_present": False}, "effect": {"decision": "deny", "reason": "asset_category_required", "required_action": "set_asset_category"}},
	{"name": "asset_requires_location", "description": "Assets require a registered location.", "condition": {"operation": "register_asset", "asset_location_present": False}, "effect": {"decision": "deny", "reason": "asset_location_required", "required_action": "attach_asset_location"}},
	{"name": "asset_requires_criticality", "description": "Assets require criticality classification.", "condition": {"operation": "register_asset", "criticality_present": False}, "effect": {"decision": "deny", "reason": "asset_criticality_required", "required_action": "classify_asset_criticality"}},
	{"name": "capital_asset_requires_fixed_asset_reference", "description": "Capitalized assets require a fixed-asset reference.", "condition": {"operation": "register_asset", "capitalized": True, "fixed_asset_ref_present": False}, "effect": {"decision": "deny", "reason": "fixed_asset_reference_required", "required_action": "attach_fixed_asset_reference"}},
	{"name": "asset_health_score_bounds_min", "description": "Asset health score cannot be below zero.", "condition": {"operation": "register_asset", "health_score_lt": 0}, "effect": {"decision": "deny", "reason": "asset_health_score_out_of_bounds", "required_action": "set_health_score_between_0_and_100"}},
	{"name": "asset_health_score_bounds_max", "description": "Asset health score cannot exceed one hundred.", "condition": {"operation": "register_asset", "health_score_gt": 100}, "effect": {"decision": "deny", "reason": "asset_health_score_out_of_bounds", "required_action": "set_health_score_between_0_and_100"}},
	{"name": "maintenance_plan_requires_strategy", "description": "Maintenance plans require a strategy.", "condition": {"operation": "create_maintenance_plan", "maintenance_strategy_present": False}, "effect": {"decision": "deny", "reason": "maintenance_strategy_required", "required_action": "set_maintenance_strategy"}},
	{"name": "maintenance_plan_requires_interval", "description": "Maintenance plans require an interval.", "condition": {"operation": "create_maintenance_plan", "interval_present": False}, "effect": {"decision": "deny", "reason": "maintenance_interval_required", "required_action": "set_maintenance_interval"}},
	{"name": "maintenance_plan_interval_positive", "description": "Maintenance plan intervals must be positive.", "condition": {"operation": "create_maintenance_plan", "interval_days_lte": 0}, "effect": {"decision": "deny", "reason": "maintenance_interval_must_be_positive", "required_action": "set_positive_interval"}},
	{"name": "predictive_plan_requires_condition_source", "description": "Predictive maintenance plans require a condition source.", "condition": {"operation": "create_maintenance_plan", "predictive_plan": True, "condition_source_present": False}, "effect": {"decision": "deny", "reason": "condition_source_required", "required_action": "attach_condition_source"}},
	{"name": "work_order_requires_asset", "description": "Work orders require an asset.", "condition": {"operation": "open_work_order", "asset_present": False}, "effect": {"decision": "deny", "reason": "work_order_asset_required", "required_action": "attach_asset"}},
	{"name": "work_order_requires_priority", "description": "Work orders require priority.", "condition": {"operation": "open_work_order", "priority_present": False}, "effect": {"decision": "deny", "reason": "work_order_priority_required", "required_action": "set_priority"}},
	{"name": "work_order_requires_safety_plan", "description": "Work orders require a safety plan.", "condition": {"operation": "open_work_order", "safety_plan_present": False}, "effect": {"decision": "deny", "reason": "work_order_safety_plan_required", "required_action": "attach_safety_plan"}},
	{"name": "critical_work_order_requires_approval", "description": "Critical work requires approval before opening.", "condition": {"operation": "open_work_order", "critical_asset": True, "approved": False}, "effect": {"decision": "require_review", "reason": "critical_work_order_approval_required", "required_action": "record_approval"}},
	{"name": "work_order_completion_requires_outcome", "description": "Work order completion requires an outcome.", "condition": {"operation": "complete_work_order", "outcome_present": False}, "effect": {"decision": "deny", "reason": "work_order_outcome_required", "required_action": "record_completion_outcome"}},
	{"name": "inspection_requires_asset", "description": "Inspections require an asset.", "condition": {"operation": "record_inspection", "asset_present": False}, "effect": {"decision": "deny", "reason": "inspection_asset_required", "required_action": "attach_asset"}},
	{"name": "inspection_requires_result", "description": "Inspections require a result.", "condition": {"operation": "record_inspection", "inspection_result_present": False}, "effect": {"decision": "deny", "reason": "inspection_result_required", "required_action": "record_inspection_result"}},
	{"name": "condition_reading_requires_metric", "description": "Condition readings require a metric.", "condition": {"operation": "record_condition_reading", "metric_present": False}, "effect": {"decision": "deny", "reason": "condition_metric_required", "required_action": "set_condition_metric"}},
	{"name": "condition_reading_requires_value", "description": "Condition readings require a value.", "condition": {"operation": "record_condition_reading", "value_present": False}, "effect": {"decision": "deny", "reason": "condition_value_required", "required_action": "set_condition_value"}},
	{"name": "condition_alert_requires_review", "description": "Alerting condition readings require review.", "condition": {"operation": "record_condition_reading", "condition_alert": True, "review_recorded": False}, "effect": {"decision": "require_review", "reason": "condition_alert_review_required", "required_action": "record_condition_review"}},
	{"name": "inventory_reservation_requires_part", "description": "Inventory reservations require a part.", "condition": {"operation": "reserve_inventory", "part_present": False}, "effect": {"decision": "deny", "reason": "inventory_part_required", "required_action": "attach_part"}},
	{"name": "inventory_reservation_requires_quantity", "description": "Inventory reservations require a quantity.", "condition": {"operation": "reserve_inventory", "quantity_present": False}, "effect": {"decision": "deny", "reason": "inventory_quantity_required", "required_action": "set_quantity"}},
	{"name": "inventory_quantity_positive", "description": "Inventory reservations must use a positive quantity.", "condition": {"operation": "reserve_inventory", "quantity_lte": 0}, "effect": {"decision": "deny", "reason": "inventory_quantity_must_be_positive", "required_action": "set_positive_quantity"}},
	{"name": "eam_batch_import_requires_bytewax", "description": "Asset batch imports require Bytewax coordination.", "condition": {"operation": "eam_batch_import", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_asset_import_to_bytewax"}},
	{"name": "eam_event_requires_bytewax", "description": "Asset lifecycle events require Bytewax.", "condition": {"operation": "eam_event", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_asset_event_to_bytewax"}},
	{"name": "eam_agent_runtime_supported", "description": "EAM agents must use an approved runtime.", "condition": {"operation": "register_eam_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "eam_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"}},
	{"name": "eam_agent_role_supported", "description": "EAM agents must use an approved role.", "condition": {"operation": "register_eam_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "eam_agent_role_not_supported", "required_action": "select_supported_agent_role"}},
	{"name": "privileged_agent_eam_action_requires_human_approval", "description": "Privileged asset actions proposed by agents require human approval.", "condition": {"operation": "agent_eam_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/eam-ast/dashboard", "component": "AssetDashboard", "permission": "eam_ast:view", "nav_group": "Overview"},
	{"name": "assets", "path": "/eam-ast/assets", "component": "AssetRegistry", "permission": "eam_ast:manage_assets", "nav_group": "Assets"},
	{"name": "locations", "path": "/eam-ast/locations", "component": "AssetLocationMap", "permission": "eam_ast:manage_locations", "nav_group": "Assets"},
	{"name": "maintenance_plans", "path": "/eam-ast/maintenance-plans", "component": "MaintenancePlanConsole", "permission": "eam_ast:manage_maintenance", "nav_group": "Maintenance"},
	{"name": "work_orders", "path": "/eam-ast/work-orders", "component": "WorkOrderConsole", "permission": "eam_ast:manage_work_orders", "nav_group": "Maintenance"},
	{"name": "inspections", "path": "/eam-ast/inspections", "component": "InspectionConsole", "permission": "eam_ast:inspect", "nav_group": "Reliability"},
	{"name": "inventory", "path": "/eam-ast/inventory", "component": "InventoryReservationConsole", "permission": "eam_ast:manage_inventory", "nav_group": "Maintenance"},
	{"name": "analytics", "path": "/eam-ast/analytics", "component": "AssetReliabilityAnalytics", "permission": "eam_ast:analytics", "nav_group": "Reliability"},
	{"name": "agents", "path": "/eam-ast/agents", "component": "EAMAgentWorkbench", "permission": "eam_ast:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/eam-ast/settings", "component": "AssetSettings", "permission": "eam_ast:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "eam_ast_control",
	"tokens": {"color.primary": "#28536B", "color.accent": "#C44536", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"},
	"components": {
		"assets": {"icon": "factory", "status_indicator": "asset-pill", "risk_style": "criticality-band"},
		"locations": {"visual": "location-tree", "status_style": "site-chip"},
		"maintenance_plans": {"visual": "plan-calendar", "status_style": "strategy-chip"},
		"work_orders": {"visual": "work-queue", "status_style": "priority-chip"},
		"inspections": {"visual": "checklist-table", "status_style": "defect-chip"},
		"inventory": {"visual": "parts-reservation-grid", "status_style": "reservation-chip"},
		"analytics": {"visual": "risk-matrix", "status_style": "health-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "eam_ast",
		"display_name": "Enterprise Asset Management",
		"provides": [
			"asset_registry_lifecycle",
			"asset_location_hierarchy",
			"criticality_and_condition_management",
			"maintenance_plan_lifecycle",
			"work_order_lifecycle",
			"inspection_and_condition_readings",
			"asset_reliability_analytics",
			"eam_agents",
		],
		"requires": ["auth", "audl", "ntfy", "composition_events", "composition_config"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/eam-ast/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": EAM_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"location_registered",
			"asset_registered",
			"maintenance_plan_created",
			"work_order_opened",
			"work_order_completed",
			"inspection_recorded",
			"condition_reading_recorded",
			"inventory_reservation_created",
			"eam_agent_registered",
		],
		"states": ["draft", "active", "in_service", "maintenance_due", "work_open", "work_complete", "degraded", "retired"],
		"guardrails": [
			"eam_batch_import_requires_bytewax",
			"eam_event_requires_bytewax",
			"privileged_agent_eam_action_requires_human_approval",
		],
	}


def event_stream_name() -> str:
	return EAM_EVENT_STREAM


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
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
