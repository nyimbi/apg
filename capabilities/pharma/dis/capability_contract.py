"""Executable capability contract for APG Pharma Distribution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_dis"
CAPABILITY_NAME = "Pharmaceutical Distribution"
CAPABILITY_VERSION = "1.0.0"
DIS_EVENT_STREAM = "apg.pharma.dis.lifecycle"

SUPPORTED_DISTRIBUTION_CHANNELS = ["wholesale", "retail_pharmacy", "hospital", "direct_to_patient", "specialty_pharmacy", "export", "humanitarian", "parallel_import"]
SUPPORTED_COLD_CHAIN_CLASSIFICATIONS = ["ambient", "controlled_room_temp", "refrigerated_2_8", "frozen_minus_20", "deep_frozen_minus_80", "cryogenic"]
SUPPORTED_SERIALISATION_STANDARDS = ["gs1_sgtin", "gs1_sscc", "dscsa", "falsified_medicines_directive", "track_and_trace", "2d_barcode", "rfid"]
SUPPORTED_RECALL_CLASSES = ["class_i", "class_ii", "class_iii", "market_withdrawal", "safety_alert"]
SUPPORTED_RECALL_STATUSES = ["initiated", "in_progress", "effectiveness_check", "completed", "terminated"]
SUPPORTED_GDP_STATUSES = ["compliant", "minor_deviation", "major_deviation", "critical_deviation", "under_corrective_action"]
SUPPORTED_WDA_STATUSES = ["applied", "granted", "conditionally_granted", "suspended", "revoked", "renewal_pending"]
SUPPORTED_TRANSPORT_MODES = ["road", "air", "sea", "rail", "courier", "cold_chain_specialist"]
SUPPORTED_SHIPMENT_STATUSES = ["planned", "picked", "dispatched", "in_transit", "customs_hold", "delivered", "exception", "recalled"]
SUPPORTED_EXCURSION_SEVERITIES = ["minor", "moderate", "major", "critical"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["cold_chain_monitor", "serialisation_verifier", "recall_coordinator", "gdp_auditor", "shipment_tracker"]
SUPPORTED_REGULATORY_MARKETS = ["eu", "us", "uk", "japan", "australia", "canada", "india", "brazil", "china", "gcc"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"distribution": {"supported_channels": SUPPORTED_DISTRIBUTION_CHANNELS, "wda_required": True, "gdp_compliance_required": True, "qualified_person_required": True},
	"cold_chain": {"supported_classifications": SUPPORTED_COLD_CHAIN_CLASSIFICATIONS, "temperature_monitoring_required": True, "excursion_reporting_required": True, "validation_required": True, "mapping_study_required": True},
	"serialisation": {"supported_standards": SUPPORTED_SERIALISATION_STANDARDS, "unique_id_required": True, "aggregation_required": True, "verification_required": True, "decommissioning_required": True},
	"recalls": {"supported_classes": SUPPORTED_RECALL_CLASSES, "supported_statuses": SUPPORTED_RECALL_STATUSES, "regulatory_notification_required": True, "effectiveness_check_required": True, "timeline_hours": {"class_i": 24, "class_ii": 72, "class_iii": 168}},
	"gdp": {"supported_statuses": SUPPORTED_GDP_STATUSES, "self_inspection_required": True, "supplier_qualification_required": True, "broker_registration_required": True},
	"wda": {"supported_statuses": SUPPORTED_WDA_STATUSES, "supported_markets": SUPPORTED_REGULATORY_MARKETS, "renewal_alert_days": 90, "scope_restriction_enforced": True},
	"shipments": {"supported_modes": SUPPORTED_TRANSPORT_MODES, "supported_statuses": SUPPORTED_SHIPMENT_STATUSES, "packing_list_required": True, "coa_required": True, "import_permit_check": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "gdp_compliance_required": True, "wda_required_for_wholesale": True, "cold_chain_breach_escalation": True, "serialisation_verification_required": True},
	"observability": {"event_stream": DIS_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_distribution": True, "enable_cold_chain": True, "enable_serialisation": True, "enable_recalls": True, "enable_gdp": True, "enable_wda": True, "enable_shipments": True},
	"theme": {"default_theme": "pharma_dis_supply", "allow_tenant_overrides": True},
}

PROVIDES = [
	"wholesale_distribution_workflow",
	"cold_chain_management_workflow",
	"serialisation_verification_workflow",
	"recall_management_workflow",
	"gdp_compliance_workflow",
	"wda_management_workflow",
	"shipment_tracking_workflow",
	"temperature_excursion_workflow",
	"import_export_workflow",
	"distribution_audit_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-dis/dashboard", "component": "DisDashboard", "permission": "pharma_dis:view", "nav_group": "Overview"},
	{"name": "shipments", "path": "/pharma-dis/shipments", "component": "ShipmentTracker", "permission": "pharma_dis:shipments", "nav_group": "Operations"},
	{"name": "shipment_detail", "path": "/pharma-dis/shipments/<id>", "component": "ShipmentDetail", "permission": "pharma_dis:shipments", "nav_group": "Operations"},
	{"name": "cold_chain", "path": "/pharma-dis/cold-chain", "component": "ColdChainMonitor", "permission": "pharma_dis:cold_chain", "nav_group": "Cold Chain"},
	{"name": "excursions", "path": "/pharma-dis/cold-chain/excursions", "component": "ExcursionLog", "permission": "pharma_dis:cold_chain", "nav_group": "Cold Chain"},
	{"name": "serialisation", "path": "/pharma-dis/serialisation", "component": "SerialisationConsole", "permission": "pharma_dis:serialisation", "nav_group": "Traceability"},
	{"name": "recalls", "path": "/pharma-dis/recalls", "component": "RecallManagement", "permission": "pharma_dis:recalls", "nav_group": "Recalls"},
	{"name": "recall_detail", "path": "/pharma-dis/recalls/<id>", "component": "RecallDetail", "permission": "pharma_dis:recalls", "nav_group": "Recalls"},
	{"name": "gdp_compliance", "path": "/pharma-dis/gdp", "component": "GdpComplianceConsole", "permission": "pharma_dis:gdp", "nav_group": "Compliance"},
	{"name": "wda", "path": "/pharma-dis/wda", "component": "WdaRegistry", "permission": "pharma_dis:wda", "nav_group": "Licensing"},
	{"name": "suppliers", "path": "/pharma-dis/suppliers", "component": "SupplierQualification", "permission": "pharma_dis:suppliers", "nav_group": "Partners"},
	{"name": "reports", "path": "/pharma-dis/reports", "component": "DistributionReports", "permission": "pharma_dis:reports", "nav_group": "Reporting"},
	{"name": "audit_trail", "path": "/pharma-dis/audit", "component": "DistributionAuditTrail", "permission": "pharma_dis:audit", "nav_group": "Compliance"},
	{"name": "settings", "path": "/pharma-dis/settings", "component": "DisSettings", "permission": "pharma_dis:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_dis_supply",
	"tokens": {
		"color.primary": "#0369A1",
		"color.accent": "#0D9488",
		"color.success": "#15803D",
		"color.warning": "#D97706",
		"color.danger": "#DC2626",
		"surface.canvas": "#F0F9FF",
		"surface.panel": "#FFFFFF",
		"text.primary": "#0C2340",
		"text.secondary": "#475569",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"shipments": {"icon": "truck", "status_indicator": "shipment-status-chip"},
		"cold_chain": {"icon": "thermometer", "status_indicator": "cold-chain-class-chip"},
		"serialisation": {"icon": "qr-code", "status_indicator": "serialisation-standard-chip"},
		"recalls": {"icon": "alert-octagon", "status_indicator": "recall-class-chip"},
		"gdp": {"icon": "shield-check", "status_indicator": "gdp-status-chip"},
		"wda": {"icon": "file-badge", "status_indicator": "wda-status-chip"},
		"suppliers": {"icon": "building", "status_indicator": "supplier-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": DIS_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"shipment_dispatched", "shipment_delivered", "shipment_exception",
		"cold_chain_excursion_detected", "temperature_breach_escalated",
		"serialisation_verified", "serialisation_violation_detected",
		"recall_initiated", "recall_completed", "gdp_deviation_recorded",
		"wda_expiring", "wda_revoked",
	],
	"guardrails": [
		"wda_required_for_wholesale_distribution",
		"cold_chain_monitoring_required",
		"serialisation_verification_required",
		"recall_timeline_enforced",
		"gdp_self_inspection_required",
		"supplier_qualification_required",
		"cross_tenant_distribution_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "wda_required_for_wholesale", "condition": {"operation": "dispatch_shipment", "channel": "wholesale", "wda_active": False}, "effect": {"decision": "deny", "reason": "wda_required_for_wholesale", "required_action": "obtain_wda"}},
	{"name": "cold_chain_classification_supported", "condition": {"operation": "create_cold_chain_record", "cold_chain_classification_supported": False}, "effect": {"decision": "deny", "reason": "cold_chain_classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "cold_chain_monitoring_required", "condition": {"operation": "dispatch_shipment", "cold_chain_product": True, "temperature_monitoring_active": False}, "effect": {"decision": "deny", "reason": "temperature_monitoring_required", "required_action": "activate_temperature_monitoring"}},
	{"name": "excursion_reporting_required", "condition": {"operation": "record_excursion", "excursion_reported": False}, "effect": {"decision": "deny", "reason": "excursion_reporting_required", "required_action": "file_excursion_report"}},
	{"name": "serialisation_standard_supported", "condition": {"operation": "serialise_product", "serialisation_standard_supported": False}, "effect": {"decision": "deny", "reason": "serialisation_standard_not_supported", "required_action": "select_supported_standard"}},
	{"name": "serialisation_verification_required", "condition": {"operation": "receive_shipment", "serialisation_verified": False}, "effect": {"decision": "deny", "reason": "serialisation_verification_required", "required_action": "verify_serialisation"}},
	{"name": "recall_class_supported", "condition": {"operation": "initiate_recall", "recall_class_supported": False}, "effect": {"decision": "deny", "reason": "recall_class_not_supported", "required_action": "select_supported_recall_class"}},
	{"name": "recall_class_i_24h", "condition": {"operation": "initiate_recall", "recall_class": "class_i", "within_24h": False}, "effect": {"decision": "deny", "reason": "class_i_recall_24h_required", "required_action": "expedite_recall_notification"}},
	{"name": "recall_regulatory_notification_required", "condition": {"operation": "initiate_recall", "regulatory_notified": False}, "effect": {"decision": "deny", "reason": "regulatory_notification_required", "required_action": "notify_regulatory_authority"}},
	{"name": "recall_effectiveness_check_required", "condition": {"operation": "close_recall", "effectiveness_check_completed": False}, "effect": {"decision": "deny", "reason": "effectiveness_check_required", "required_action": "complete_effectiveness_check"}},
	{"name": "gdp_supplier_qualification_required", "condition": {"operation": "add_supplier", "supplier_qualified": False}, "effect": {"decision": "deny", "reason": "supplier_qualification_required", "required_action": "qualify_supplier"}},
	{"name": "shipment_packing_list_required", "condition": {"operation": "dispatch_shipment", "packing_list_present": False}, "effect": {"decision": "deny", "reason": "packing_list_required", "required_action": "attach_packing_list"}},
	{"name": "shipment_coa_required", "condition": {"operation": "dispatch_shipment", "coa_present": False}, "effect": {"decision": "deny", "reason": "coa_required", "required_action": "attach_coa"}},
	{"name": "import_permit_check_required", "condition": {"operation": "import_shipment", "import_permit_checked": False}, "effect": {"decision": "deny", "reason": "import_permit_check_required", "required_action": "verify_import_permit"}},
	{"name": "wda_renewal_alert", "condition": {"operation": "check_wda", "wda_expiring_within_90d": True, "renewal_initiated": False}, "effect": {"decision": "deny", "reason": "wda_renewal_required", "required_action": "initiate_wda_renewal"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
	{"name": "gdp_self_inspection_required", "condition": {"operation": "complete_gdp_cycle", "self_inspection_completed": False}, "effect": {"decision": "deny", "reason": "gdp_self_inspection_required", "required_action": "schedule_self_inspection"}},
	{"name": "transport_mode_supported", "condition": {"operation": "dispatch_shipment", "transport_mode_supported": False}, "effect": {"decision": "deny", "reason": "transport_mode_not_supported", "required_action": "select_supported_transport_mode"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"provides": list(PROVIDES),
		"requires": list(REQUIRES),
		"configuration": configuration,
		"configuration_schema": {
			"type": "object",
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string", "minLength": 1},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
			},
		},
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/pharma-dis/api/v1",
			"requires_theme": True,
			"template_roots": ["templates/", "static/"],
			"routes": deepcopy(UI_ROUTES),
		},
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
