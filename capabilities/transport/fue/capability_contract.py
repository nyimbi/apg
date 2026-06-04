"""Executable capability contract for APG Fuel Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_fue"
CAPABILITY_NAME = "Fuel Management"
CAPABILITY_VERSION = "1.0.0"
FUEL_EVENT_STREAM = "apg.transport.fuel.lifecycle"

SUPPORTED_FUEL_TYPES = ["diesel", "petrol", "cng", "lng", "electric", "hybrid", "hydrogen", "biodiesel", "adblue", "aviation_fuel", "marine_fuel"]
SUPPORTED_PROCUREMENT_TYPES = ["spot_purchase", "contract_supply", "fuel_card", "bulk_storage", "bunker_supply", "on_road_purchase"]
SUPPORTED_TRANSACTION_TYPES = ["fill_up", "partial_fill", "def_top_up", "lubricant_purchase", "additive_purchase", "bunker_delivery"]
SUPPORTED_CARD_PROVIDERS = ["shell", "bp", "total", "esso", "keyfuels", "uk_fuels", "allstar", "arval", "uta", "dkv", "euroshell", "as24"]
SUPPORTED_CARBON_STANDARDS = ["ghg_protocol", "iso14064", "defra", "unfccc", "eu_ets", "sasb_tr", "tcfd"]
SUPPORTED_EFFICIENCY_METRICS = ["l_per_100km", "mpg_uk", "mpg_us", "km_per_litre", "kwh_per_100km", "g_co2_per_km"]
SUPPORTED_STORAGE_TYPES = ["above_ground_tank", "underground_tank", "mobile_bowser", "depot_pump", "bunkered_storage"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["fuel_procurement_agent", "consumption_analyst", "carbon_reporter", "card_reconciler", "bunker_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"fuel_types": {"supported_types": SUPPORTED_FUEL_TYPES, "unit_of_measure": "litres", "adblue_tracking_enabled": True},
	"procurement": {"types": SUPPORTED_PROCUREMENT_TYPES, "supplier_required": True, "purchase_order_required": True, "bulk_discount_tracking": True},
	"transactions": {"types": SUPPORTED_TRANSACTION_TYPES, "vehicle_required": True, "driver_required": True, "odometer_required": True, "receipt_capture_required": True},
	"fuel_cards": {"providers": SUPPORTED_CARD_PROVIDERS, "reconciliation_frequency": "daily", "fraud_detection_enabled": True, "pin_required": True},
	"carbon": {"standards": SUPPORTED_CARBON_STANDARDS, "emission_factors_updated_annually": True, "scope1_reporting": True, "scope3_enabled": False, "net_zero_target_tracking": True},
	"efficiency": {"metrics": SUPPORTED_EFFICIENCY_METRICS, "benchmarking_enabled": True, "alert_on_anomaly": True, "idle_fuel_tracking": True},
	"storage": {"types": SUPPORTED_STORAGE_TYPES, "level_monitoring_enabled": True, "spill_alert_enabled": True, "calibration_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_fuel_denied": True, "fuel_theft_alert_enabled": True, "phantom_fill_detection": True},
	"observability": {"event_stream": FUEL_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_procurement": True, "enable_transactions": True, "enable_fuel_cards": True, "enable_carbon": True, "enable_storage": True},
	"theme": {"default_theme": "transport_fuel_control", "allow_tenant_overrides": True},
}

PROVIDES = ["fuel_procurement_workflow", "fuel_consumption_tracking_workflow", "bunker_management_workflow", "fuel_card_reconciliation_workflow", "carbon_footprint_reporting_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-fuel/dashboard", "component": "FuelDashboard", "permission": "transport_fue:view", "nav_group": "Overview"},
	{"name": "procurement", "path": "/transport-fuel/procurement", "component": "FuelProcurementConsole", "permission": "transport_fue:procurement", "nav_group": "Procurement"},
	{"name": "transactions", "path": "/transport-fuel/transactions", "component": "FuelTransactionConsole", "permission": "transport_fue:transactions", "nav_group": "Transactions"},
	{"name": "fuel_cards", "path": "/transport-fuel/cards", "component": "FuelCardConsole", "permission": "transport_fue:cards", "nav_group": "Cards"},
	{"name": "card_reconciliation", "path": "/transport-fuel/cards/reconciliation", "component": "FuelCardReconciliation", "permission": "transport_fue:cards", "nav_group": "Cards"},
	{"name": "storage", "path": "/transport-fuel/storage", "component": "FuelStorageConsole", "permission": "transport_fue:storage", "nav_group": "Storage"},
	{"name": "efficiency", "path": "/transport-fuel/efficiency", "component": "FuelEfficiencyConsole", "permission": "transport_fue:efficiency", "nav_group": "Analytics"},
	{"name": "carbon", "path": "/transport-fuel/carbon", "component": "CarbonFootprintConsole", "permission": "transport_fue:carbon", "nav_group": "Sustainability"},
	{"name": "suppliers", "path": "/transport-fuel/suppliers", "component": "FuelSupplierConsole", "permission": "transport_fue:suppliers", "nav_group": "Procurement"},
	{"name": "reports", "path": "/transport-fuel/reports", "component": "FuelReportConsole", "permission": "transport_fue:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-fuel/agents", "component": "FuelAgentWorkbench", "permission": "transport_fue:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-fuel/settings", "component": "FuelSettings", "permission": "transport_fue:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_fuel_control",
	"tokens": {"color.primary": "#B45309", "color.accent": "#DC2626", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#FFFBEB", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "6px", "density": "comfortable"},
	"components": {
		"procurement": {"icon": "shopping-cart", "status_indicator": "procurement-type-chip"},
		"transactions": {"icon": "fuel", "status_indicator": "transaction-type-chip"},
		"fuel_cards": {"icon": "credit-card", "status_indicator": "card-provider-chip"},
		"storage": {"icon": "database", "status_indicator": "storage-type-chip"},
		"efficiency": {"icon": "trending-up", "status_indicator": "efficiency-metric-chip"},
		"carbon": {"icon": "leaf", "status_indicator": "carbon-standard-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": FUEL_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["fuel_procurement_recorded", "fuel_transaction_recorded", "fuel_card_reconciled", "carbon_emission_calculated", "fuel_storage_updated", "efficiency_alert_raised", "fuel_theft_detected", "fuel_agent_registered"],
	"guardrails": ["fuel_batch_requires_bytewax", "cross_tenant_fuel_denied", "fuel_theft_alert_enabled", "phantom_fill_detection", "privileged_fuel_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "fuel_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "fuel_policy_required", "required_action": "attach_fuel_policy"}},
	{"name": "fuel_type_supported", "condition": {"operation": "record_transaction", "fuel_type_supported": False}, "effect": {"decision": "deny", "reason": "fuel_type_not_supported", "required_action": "select_supported_fuel_type"}},
	{"name": "transaction_vehicle_required", "condition": {"operation": "record_transaction", "vehicle_present": False}, "effect": {"decision": "deny", "reason": "vehicle_required", "required_action": "assign_vehicle"}},
	{"name": "transaction_driver_required", "condition": {"operation": "record_transaction", "driver_present": False}, "effect": {"decision": "deny", "reason": "driver_required", "required_action": "assign_driver"}},
	{"name": "transaction_odometer_required", "condition": {"operation": "record_transaction", "odometer_present": False}, "effect": {"decision": "deny", "reason": "odometer_reading_required", "required_action": "provide_odometer_reading"}},
	{"name": "transaction_type_supported", "condition": {"operation": "record_transaction", "transaction_type_supported": False}, "effect": {"decision": "deny", "reason": "transaction_type_not_supported", "required_action": "select_supported_transaction_type"}},
	{"name": "transaction_quantity_positive", "condition": {"operation": "record_transaction", "quantity_positive": False}, "effect": {"decision": "deny", "reason": "fuel_quantity_must_be_positive", "required_action": "correct_fuel_quantity"}},
	{"name": "phantom_fill_detection", "condition": {"operation": "record_transaction", "phantom_fill_detected": True}, "effect": {"decision": "deny", "reason": "phantom_fill_detected", "required_action": "investigate_suspicious_transaction"}},
	{"name": "procurement_supplier_required", "condition": {"operation": "create_procurement", "supplier_present": False}, "effect": {"decision": "deny", "reason": "supplier_required", "required_action": "assign_supplier"}},
	{"name": "procurement_type_supported", "condition": {"operation": "create_procurement", "procurement_type_supported": False}, "effect": {"decision": "deny", "reason": "procurement_type_not_supported", "required_action": "select_supported_procurement_type"}},
	{"name": "fuel_card_provider_supported", "condition": {"operation": "register_fuel_card", "provider_supported": False}, "effect": {"decision": "deny", "reason": "card_provider_not_supported", "required_action": "select_supported_provider"}},
	{"name": "carbon_standard_supported", "condition": {"operation": "record_carbon_emission", "standard_supported": False}, "effect": {"decision": "deny", "reason": "carbon_standard_not_supported", "required_action": "select_supported_standard"}},
	{"name": "storage_type_supported", "condition": {"operation": "register_storage", "storage_type_supported": False}, "effect": {"decision": "deny", "reason": "storage_type_not_supported", "required_action": "select_supported_storage_type"}},
	{"name": "efficiency_metric_supported", "condition": {"operation": "record_efficiency", "metric_supported": False}, "effect": {"decision": "deny", "reason": "efficiency_metric_not_supported", "required_action": "select_supported_metric"}},
	{"name": "cross_tenant_fuel_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_fuel_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "fuel_batch_requires_bytewax", "condition": {"operation": "fuel_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_fuel_batch_to_bytewax"}},
	{"name": "fuel_agent_runtime_supported", "condition": {"operation": "register_fuel_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "fuel_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "fuel_agent_role_supported", "condition": {"operation": "register_fuel_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "fuel_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_fuel_agent_action_requires_human_approval", "condition": {"operation": "fuel_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "fuel_theft_alert_enabled", "condition": {"operation": "record_transaction", "theft_pattern_detected": True}, "effect": {"decision": "deny", "reason": "fuel_theft_pattern_detected", "required_action": "flag_transaction_for_investigation"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"name": CAPABILITY_NAME,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {"shell": "apg_python", "api_prefix": "/transport-fuel/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
		"theme": deepcopy(THEME),
		"streaming": deepcopy(STREAMING),
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	actions: list[dict[str, Any]] = []
	for rule in RULES:
		if _matches(rule["condition"], context):
			actions.append(rule["effect"] | {"rule": rule["name"]})
	if not actions:
		return {"decision": "allow", "actions": [], "context": dict(context)}
	return {"decision": "deny", "actions": actions, "context": dict(context)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
			continue
		if context.get(key) != expected:
			return False
	return True
