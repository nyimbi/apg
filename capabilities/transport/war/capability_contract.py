"""Executable capability contract for APG Warehouse Operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_war"
CAPABILITY_NAME = "Warehouse Operations"
CAPABILITY_VERSION = "1.0.0"
WAREHOUSE_EVENT_STREAM = "apg.transport.warehouse.lifecycle"

SUPPORTED_WAREHOUSE_TYPES = ["ambient", "temperature_controlled", "cold_store", "bonded", "hazmat", "high_security", "cross_dock", "fulfillment_centre", "distribution_centre", "dark_store"]
SUPPORTED_RECEIPT_METHODS = ["asn_based", "blind_receipt", "po_based", "return_receipt", "transfer_receipt", "cross_dock_receipt"]
SUPPORTED_PUTAWAY_STRATEGIES = ["fixed_location", "random_location", "zone_based", "abc_velocity", "bulk_to_pick", "directed_putaway", "clustering"]
SUPPORTED_PICK_METHODS = ["single_order", "batch_pick", "zone_pick", "wave_pick", "cluster_pick", "voice_directed", "scan_to_pick", "robotic_pick"]
SUPPORTED_PACK_TYPES = ["standard_carton", "custom_carton", "pallet", "polybag", "envelope", "tube", "crate", "blister_pack", "gift_wrap"]
SUPPORTED_CYCLE_COUNT_TYPES = ["abc_analysis", "random_sample", "location_based", "movement_based", "full_count", "spot_check"]
SUPPORTED_DOCK_DOOR_STATUSES = ["available", "occupied_inbound", "occupied_outbound", "reserved", "maintenance", "blocked"]
SUPPORTED_STORAGE_CONDITIONS = ["ambient", "chilled_2_8", "frozen_minus_18", "controlled_room_temp", "flammable_store", "pharmaceutical", "high_value"]
SUPPORTED_WMS_INTEGRATION_TYPES = ["sap_ewm", "oracle_wms", "manhattan_wms", "blue_yonder", "infor_wms", "korber", "custom_api", "flat_file"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["receiving_agent", "putaway_coordinator", "pick_optimiser", "packing_agent", "inventory_auditor", "cross_dock_manager"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"warehouses": {"supported_types": SUPPORTED_WAREHOUSE_TYPES, "storage_conditions": SUPPORTED_STORAGE_CONDITIONS, "dock_door_management": True, "yard_management_enabled": True},
	"receiving": {"methods": SUPPORTED_RECEIPT_METHODS, "barcode_scan_required": True, "damage_inspection_required": True, "temperature_check_for_cold_chain": True},
	"putaway": {"strategies": SUPPORTED_PUTAWAY_STRATEGIES, "default_strategy": "zone_based", "slot_verification_required": True, "confirmation_scan_required": True},
	"picking": {"methods": SUPPORTED_PICK_METHODS, "default_method": "single_order", "priority_queue_enabled": True, "short_pick_handling": True},
	"packing": {"pack_types": SUPPORTED_PACK_TYPES, "weight_check_required": True, "dim_weight_calculation": True, "packing_slip_required": True, "label_print_required": True},
	"cycle_counting": {"types": SUPPORTED_CYCLE_COUNT_TYPES, "default_type": "abc_analysis", "discrepancy_threshold_pct": 1.0, "auto_adjust_enabled": False, "approval_required_for_adjust": True},
	"dock_doors": {"statuses": SUPPORTED_DOCK_DOOR_STATUSES, "appointment_scheduling_enabled": True, "dwell_time_tracking": True},
	"wms_integration": {"types": SUPPORTED_WMS_INTEGRATION_TYPES, "bidirectional_sync": True, "real_time_inventory_enabled": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_warehouse_denied": True, "inventory_manipulation_denied": True, "unapproved_stock_adjustment_denied": True},
	"observability": {"event_stream": WAREHOUSE_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_receiving": True, "enable_putaway": True, "enable_picking": True, "enable_packing": True, "enable_cycle_counting": True},
	"theme": {"default_theme": "transport_warehouse_control", "allow_tenant_overrides": True},
}

PROVIDES = ["warehouse_receiving_workflow", "putaway_workflow", "picking_workflow", "packing_workflow", "cross_docking_workflow", "cycle_counting_workflow", "wms_integration_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-warehouse/dashboard", "component": "WarehouseDashboard", "permission": "transport_war:view", "nav_group": "Overview"},
	{"name": "receiving", "path": "/transport-warehouse/receiving", "component": "ReceivingConsole", "permission": "transport_war:receiving", "nav_group": "Inbound"},
	{"name": "putaway", "path": "/transport-warehouse/putaway", "component": "PutawayConsole", "permission": "transport_war:putaway", "nav_group": "Inbound"},
	{"name": "inventory", "path": "/transport-warehouse/inventory", "component": "InventoryConsole", "permission": "transport_war:inventory", "nav_group": "Inventory"},
	{"name": "picking", "path": "/transport-warehouse/picking", "component": "PickingConsole", "permission": "transport_war:picking", "nav_group": "Outbound"},
	{"name": "packing", "path": "/transport-warehouse/packing", "component": "PackingConsole", "permission": "transport_war:packing", "nav_group": "Outbound"},
	{"name": "cross_docking", "path": "/transport-warehouse/cross-dock", "component": "CrossDockConsole", "permission": "transport_war:cross_dock", "nav_group": "Operations"},
	{"name": "cycle_counting", "path": "/transport-warehouse/cycle-count", "component": "CycleCountConsole", "permission": "transport_war:cycle_count", "nav_group": "Inventory"},
	{"name": "dock_doors", "path": "/transport-warehouse/dock-doors", "component": "DockDoorConsole", "permission": "transport_war:dock_doors", "nav_group": "Yard"},
	{"name": "wms_integration", "path": "/transport-warehouse/wms", "component": "WmsIntegrationConsole", "permission": "transport_war:wms", "nav_group": "Integration"},
	{"name": "reports", "path": "/transport-warehouse/reports", "component": "WarehouseReportConsole", "permission": "transport_war:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-warehouse/agents", "component": "WarehouseAgentWorkbench", "permission": "transport_war:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-warehouse/settings", "component": "WarehouseSettings", "permission": "transport_war:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_warehouse_control",
	"tokens": {"color.primary": "#374151", "color.accent": "#0369A1", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F9FAFB", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "6px", "density": "compact"},
	"components": {
		"receiving": {"icon": "inbox", "status_indicator": "receipt-method-chip"},
		"putaway": {"icon": "arrow-down-circle", "status_indicator": "putaway-strategy-chip"},
		"picking": {"icon": "hand", "status_indicator": "pick-method-chip"},
		"packing": {"icon": "package", "status_indicator": "pack-type-chip"},
		"cycle_counting": {"icon": "list-ordered", "status_indicator": "count-type-chip"},
		"dock_doors": {"icon": "door-open", "status_indicator": "dock-status-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": WAREHOUSE_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["goods_received", "putaway_completed", "pick_task_created", "pick_completed", "packing_completed", "cross_dock_completed", "cycle_count_completed", "inventory_adjusted", "dock_door_allocated", "warehouse_agent_registered"],
	"guardrails": ["warehouse_batch_requires_bytewax", "unapproved_stock_adjustment_denied", "inventory_manipulation_denied", "cross_tenant_warehouse_denied", "privileged_warehouse_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "warehouse_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "warehouse_policy_required", "required_action": "attach_warehouse_policy"}},
	{"name": "warehouse_type_supported", "condition": {"operation": "register_warehouse", "warehouse_type_supported": False}, "effect": {"decision": "deny", "reason": "warehouse_type_not_supported", "required_action": "select_supported_warehouse_type"}},
	{"name": "receipt_method_supported", "condition": {"operation": "receive_goods", "receipt_method_supported": False}, "effect": {"decision": "deny", "reason": "receipt_method_not_supported", "required_action": "select_supported_receipt_method"}},
	{"name": "receipt_barcode_required", "condition": {"operation": "receive_goods", "barcode_scanned": False}, "effect": {"decision": "deny", "reason": "barcode_scan_required_for_receiving", "required_action": "scan_item_barcode"}},
	{"name": "receipt_damage_inspection_required", "condition": {"operation": "receive_goods", "damage_inspection_completed": False}, "effect": {"decision": "deny", "reason": "damage_inspection_required", "required_action": "complete_damage_inspection"}},
	{"name": "putaway_strategy_supported", "condition": {"operation": "execute_putaway", "strategy_supported": False}, "effect": {"decision": "deny", "reason": "putaway_strategy_not_supported", "required_action": "select_supported_putaway_strategy"}},
	{"name": "putaway_slot_verification_required", "condition": {"operation": "execute_putaway", "slot_verified": False}, "effect": {"decision": "deny", "reason": "putaway_slot_verification_required", "required_action": "verify_putaway_slot"}},
	{"name": "pick_method_supported", "condition": {"operation": "create_pick_task", "pick_method_supported": False}, "effect": {"decision": "deny", "reason": "pick_method_not_supported", "required_action": "select_supported_pick_method"}},
	{"name": "pack_type_supported", "condition": {"operation": "create_pack_task", "pack_type_supported": False}, "effect": {"decision": "deny", "reason": "pack_type_not_supported", "required_action": "select_supported_pack_type"}},
	{"name": "packing_weight_required", "condition": {"operation": "complete_packing", "weight_checked": False}, "effect": {"decision": "deny", "reason": "weight_check_required_before_dispatch", "required_action": "weigh_packed_shipment"}},
	{"name": "cycle_count_type_supported", "condition": {"operation": "initiate_cycle_count", "count_type_supported": False}, "effect": {"decision": "deny", "reason": "cycle_count_type_not_supported", "required_action": "select_supported_count_type"}},
	{"name": "unapproved_stock_adjustment_denied", "condition": {"operation": "adjust_inventory", "approval_present": False}, "effect": {"decision": "deny", "reason": "unapproved_stock_adjustment_denied", "required_action": "obtain_adjustment_approval"}},
	{"name": "inventory_manipulation_denied", "condition": {"operation": "adjust_inventory", "manipulation_detected": True}, "effect": {"decision": "deny", "reason": "inventory_manipulation_denied", "required_action": "investigate_inventory_discrepancy"}},
	{"name": "dock_door_status_supported", "condition": {"operation": "update_dock_door_status", "status_supported": False}, "effect": {"decision": "deny", "reason": "dock_door_status_not_supported", "required_action": "select_supported_dock_status"}},
	{"name": "wms_integration_type_supported", "condition": {"operation": "configure_wms_integration", "integration_type_supported": False}, "effect": {"decision": "deny", "reason": "wms_integration_type_not_supported", "required_action": "select_supported_integration_type"}},
	{"name": "cross_tenant_warehouse_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_warehouse_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "warehouse_batch_requires_bytewax", "condition": {"operation": "warehouse_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_warehouse_batch_to_bytewax"}},
	{"name": "warehouse_agent_runtime_supported", "condition": {"operation": "register_warehouse_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "warehouse_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "warehouse_agent_role_supported", "condition": {"operation": "register_warehouse_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "warehouse_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_warehouse_agent_action_requires_human_approval", "condition": {"operation": "warehouse_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
	{"name": "cold_chain_temp_check_required", "condition": {"operation": "receive_goods", "cold_chain_required": True, "temperature_checked": False}, "effect": {"decision": "deny", "reason": "temperature_check_required_for_cold_chain", "required_action": "record_inbound_temperature"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-warehouse/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
