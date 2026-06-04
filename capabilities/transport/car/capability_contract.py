"""Executable capability contract for APG Cargo Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "transport_car"
CAPABILITY_NAME = "Cargo Management"
CAPABILITY_VERSION = "1.0.0"
CARGO_EVENT_STREAM = "apg.transport.cargo.lifecycle"

SUPPORTED_CARGO_TYPES = ["general", "bulk", "liquid", "refrigerated", "frozen", "hazardous", "oversized", "fragile", "livestock", "valuable", "pharmaceutical", "automotive"]
SUPPORTED_DG_CLASSES = ["class_1_explosives", "class_2_gases", "class_3_flammable_liquids", "class_4_flammable_solids", "class_5_oxidizers", "class_6_toxic", "class_7_radioactive", "class_8_corrosives", "class_9_miscellaneous"]
SUPPORTED_BOOKING_STATUSES = ["draft", "confirmed", "in_transit", "delivered", "cancelled", "on_hold"]
SUPPORTED_MANIFEST_STATUSES = ["draft", "submitted", "accepted", "rejected", "amended"]
SUPPORTED_TRACKING_EVENTS = ["booked", "collected", "in_transit", "customs_hold", "out_for_delivery", "delivered", "exception", "returned"]
SUPPORTED_REVENUE_TYPES = ["freight_charge", "fuel_surcharge", "hazmat_surcharge", "oversize_surcharge", "storage_fee", "customs_fee", "insurance_fee", "handling_fee"]
SUPPORTED_COMPLIANCE_STANDARDS = ["iata", "imdg", "adr", "rid", "adnr", "icao", "c_tpat", "ctpat_plus"]
SUPPORTED_PACKAGING_TYPES = ["pallet", "crate", "drum", "ibc", "flexibag", "container_20ft", "container_40ft", "container_reefer", "loose", "roll", "coil"]
SUPPORTED_INCOTERMS = ["exw", "fca", "cpт", "cip", "dat", "dap", "ddp", "fas", "fob", "cfr", "cif"]
SUPPORTED_WEIGHT_UNITS = ["kg", "lb", "ton_metric", "ton_imperial"]
SUPPORTED_VOLUME_UNITS = ["cbm", "cft", "litre"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["booking_agent", "manifest_steward", "compliance_checker", "revenue_optimizer", "tracking_monitor"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"cargo_types": {"supported_types": SUPPORTED_CARGO_TYPES, "dg_classes": SUPPORTED_DG_CLASSES, "packaging_types": SUPPORTED_PACKAGING_TYPES, "weight_units": SUPPORTED_WEIGHT_UNITS, "volume_units": SUPPORTED_VOLUME_UNITS},
	"bookings": {"supported_statuses": SUPPORTED_BOOKING_STATUSES, "incoterms": SUPPORTED_INCOTERMS, "shipper_required": True, "consignee_required": True, "origin_required": True, "destination_required": True, "weight_required": True},
	"manifests": {"supported_statuses": SUPPORTED_MANIFEST_STATUSES, "booking_required": True, "customs_declaration_required": True, "dg_declaration_required_for_hazmat": True},
	"dangerous_goods": {"compliance_standards": SUPPORTED_COMPLIANCE_STANDARDS, "dg_class_required": True, "un_number_required": True, "packing_group_required": True, "emergency_contact_required": True},
	"tracking": {"supported_events": SUPPORTED_TRACKING_EVENTS, "real_time_enabled": True, "geofencing_enabled": True, "iot_integration": True},
	"revenue": {"supported_types": SUPPORTED_REVENUE_TYPES, "currency_required": True, "rate_card_required": True, "approval_required_above_threshold": True},
	"compliance": {"standards": SUPPORTED_COMPLIANCE_STANDARDS, "audit_required": True, "certificate_required": True, "evidence_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "cross_tenant_cargo_denied": True, "unapproved_dg_shipment_denied": True, "weight_falsification_denied": True},
	"observability": {"event_stream": CARGO_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_bookings": True, "enable_manifests": True, "enable_dangerous_goods": True, "enable_tracking": True, "enable_revenue": True, "enable_compliance": True},
	"theme": {"default_theme": "transport_cargo_control", "allow_tenant_overrides": True},
}

PROVIDES = ["cargo_booking_workflow", "cargo_manifest_workflow", "dangerous_goods_compliance_workflow", "cargo_tracking_workflow", "cargo_revenue_workflow", "cargo_compliance_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "comp", "mqeb", "schd"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/transport-cargo/dashboard", "component": "CargoDashboard", "permission": "transport_car:view", "nav_group": "Overview"},
	{"name": "bookings", "path": "/transport-cargo/bookings", "component": "CargoBookingConsole", "permission": "transport_car:bookings", "nav_group": "Bookings"},
	{"name": "booking_create", "path": "/transport-cargo/bookings/create", "component": "CargoBookingForm", "permission": "transport_car:bookings_write", "nav_group": "Bookings"},
	{"name": "manifests", "path": "/transport-cargo/manifests", "component": "CargoManifestConsole", "permission": "transport_car:manifests", "nav_group": "Documentation"},
	{"name": "dangerous_goods", "path": "/transport-cargo/dangerous-goods", "component": "DangerousGoodsConsole", "permission": "transport_car:dg_compliance", "nav_group": "Compliance"},
	{"name": "tracking", "path": "/transport-cargo/tracking", "component": "CargoTrackingBoard", "permission": "transport_car:tracking", "nav_group": "Operations"},
	{"name": "tracking_detail", "path": "/transport-cargo/tracking/<booking_id>", "component": "CargoTrackingDetail", "permission": "transport_car:tracking", "nav_group": "Operations"},
	{"name": "revenue", "path": "/transport-cargo/revenue", "component": "CargoRevenueConsole", "permission": "transport_car:revenue", "nav_group": "Finance"},
	{"name": "compliance", "path": "/transport-cargo/compliance", "component": "CargoComplianceConsole", "permission": "transport_car:compliance", "nav_group": "Compliance"},
	{"name": "rate_cards", "path": "/transport-cargo/rate-cards", "component": "CargoRateCardConsole", "permission": "transport_car:revenue", "nav_group": "Finance"},
	{"name": "reports", "path": "/transport-cargo/reports", "component": "CargoReportConsole", "permission": "transport_car:reports", "nav_group": "Reporting"},
	{"name": "agents", "path": "/transport-cargo/agents", "component": "CargoAgentWorkbench", "permission": "transport_car:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/transport-cargo/settings", "component": "CargoSettings", "permission": "transport_car:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "transport_cargo_control",
	"tokens": {"color.primary": "#1E40AF", "color.accent": "#0891B2", "color.success": "#166534", "color.warning": "#A16207", "color.danger": "#991B1B", "surface.canvas": "#F0F9FF", "surface.panel": "#FFFFFF", "text.primary": "#0F172A", "text.secondary": "#475569", "border.radius": "6px", "density": "comfortable"},
	"components": {
		"bookings": {"icon": "package", "status_indicator": "booking-status-chip"},
		"manifests": {"icon": "file-text", "status_indicator": "manifest-status-chip"},
		"dangerous_goods": {"icon": "alert-triangle", "status_indicator": "dg-class-chip"},
		"tracking": {"icon": "map-pin", "status_indicator": "tracking-event-chip"},
		"revenue": {"icon": "dollar-sign", "status_indicator": "revenue-type-chip"},
		"compliance": {"icon": "shield", "status_indicator": "compliance-chip"},
		"agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CARGO_EVENT_STREAM,
	"key": "tenant_id",
	"events": ["cargo_booked", "cargo_manifest_submitted", "cargo_dg_declared", "cargo_tracking_updated", "cargo_delivered", "cargo_revenue_recorded", "cargo_compliance_checked", "cargo_agent_registered"],
	"guardrails": ["cargo_batch_requires_bytewax", "unapproved_dg_shipment_denied", "weight_falsification_denied", "cross_tenant_cargo_denied", "privileged_cargo_agent_action_requires_human_approval"],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "cargo_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "cargo_policy_required", "required_action": "attach_cargo_policy"}},
	{"name": "booking_shipper_required", "condition": {"operation": "create_booking", "shipper_present": False}, "effect": {"decision": "deny", "reason": "shipper_required", "required_action": "attach_shipper_reference"}},
	{"name": "booking_consignee_required", "condition": {"operation": "create_booking", "consignee_present": False}, "effect": {"decision": "deny", "reason": "consignee_required", "required_action": "attach_consignee_reference"}},
	{"name": "booking_origin_required", "condition": {"operation": "create_booking", "origin_present": False}, "effect": {"decision": "deny", "reason": "origin_required", "required_action": "set_origin_location"}},
	{"name": "booking_destination_required", "condition": {"operation": "create_booking", "destination_present": False}, "effect": {"decision": "deny", "reason": "destination_required", "required_action": "set_destination_location"}},
	{"name": "booking_weight_required", "condition": {"operation": "create_booking", "weight_present": False}, "effect": {"decision": "deny", "reason": "weight_required", "required_action": "declare_cargo_weight"}},
	{"name": "booking_cargo_type_supported", "condition": {"operation": "create_booking", "cargo_type_supported": False}, "effect": {"decision": "deny", "reason": "cargo_type_not_supported", "required_action": "select_supported_cargo_type"}},
	{"name": "manifest_booking_required", "condition": {"operation": "create_manifest", "booking_present": False}, "effect": {"decision": "deny", "reason": "booking_required", "required_action": "select_booking"}},
	{"name": "manifest_status_supported", "condition": {"operation": "create_manifest", "manifest_status_supported": False}, "effect": {"decision": "deny", "reason": "manifest_status_not_supported", "required_action": "select_supported_manifest_status"}},
	{"name": "dg_un_number_required", "condition": {"operation": "declare_dangerous_goods", "un_number_present": False}, "effect": {"decision": "deny", "reason": "un_number_required", "required_action": "provide_un_number"}},
	{"name": "dg_class_required", "condition": {"operation": "declare_dangerous_goods", "dg_class_present": False}, "effect": {"decision": "deny", "reason": "dg_class_required", "required_action": "select_dg_class"}},
	{"name": "dg_packing_group_required", "condition": {"operation": "declare_dangerous_goods", "packing_group_present": False}, "effect": {"decision": "deny", "reason": "packing_group_required", "required_action": "select_packing_group"}},
	{"name": "dg_emergency_contact_required", "condition": {"operation": "declare_dangerous_goods", "emergency_contact_present": False}, "effect": {"decision": "deny", "reason": "emergency_contact_required", "required_action": "provide_emergency_contact"}},
	{"name": "unapproved_dg_shipment_denied", "condition": {"operation": "create_booking", "cargo_type": "hazardous", "dg_approved": False}, "effect": {"decision": "deny", "reason": "dg_approval_required", "required_action": "obtain_dg_approval"}},
	{"name": "tracking_event_type_supported", "condition": {"operation": "update_tracking", "tracking_event_supported": False}, "effect": {"decision": "deny", "reason": "tracking_event_not_supported", "required_action": "select_supported_tracking_event"}},
	{"name": "tracking_location_required", "condition": {"operation": "update_tracking", "location_present": False}, "effect": {"decision": "deny", "reason": "tracking_location_required", "required_action": "provide_tracking_location"}},
	{"name": "revenue_type_supported", "condition": {"operation": "record_revenue", "revenue_type_supported": False}, "effect": {"decision": "deny", "reason": "revenue_type_not_supported", "required_action": "select_supported_revenue_type"}},
	{"name": "revenue_currency_required", "condition": {"operation": "record_revenue", "currency_present": False}, "effect": {"decision": "deny", "reason": "currency_required", "required_action": "specify_currency"}},
	{"name": "revenue_amount_positive", "condition": {"operation": "record_revenue", "amount_positive": False}, "effect": {"decision": "deny", "reason": "revenue_amount_must_be_positive", "required_action": "correct_revenue_amount"}},
	{"name": "cross_tenant_cargo_denied", "condition": {"operation_type": "write", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_cargo_denied", "required_action": "use_tenant_scoped_context"}},
	{"name": "weight_falsification_denied", "condition": {"operation": "create_booking", "weight_falsification_detected": True}, "effect": {"decision": "deny", "reason": "weight_falsification_denied", "required_action": "provide_accurate_weight"}},
	{"name": "cargo_batch_requires_bytewax", "condition": {"operation": "cargo_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_cargo_batch_to_bytewax"}},
	{"name": "cargo_agent_runtime_supported", "condition": {"operation": "register_cargo_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "cargo_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "cargo_agent_role_supported", "condition": {"operation": "register_cargo_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "cargo_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "privileged_cargo_agent_action_requires_human_approval", "condition": {"operation": "cargo_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
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
		"ui": {"shell": "apg_python", "api_prefix": "/transport-cargo/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)},
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
