"""Executable capability contract for APG Ore Processing & Metallurgy."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mining_ore"
CAPABILITY_NAME = "Ore Processing & Metallurgy"
CAPABILITY_VERSION = "1.0.0"
ORE_EVENT_STREAM = "apg.mining.ore.lifecycle"

SUPPORTED_PROCESS_CIRCUITS = ["primary_crushing", "secondary_crushing", "tertiary_crushing", "sag_milling", "ball_milling", "flotation", "leaching", "cil", "cip", "heap_leach", "sxew", "gravity_concentration", "magnetic_separation", "dense_media_separation", "thickening", "filtration", "drying", "smelting", "refining"]
SUPPORTED_REAGENT_TYPES = ["cyanide", "lime", "sulphuric_acid", "xanthate", "frother", "flocculant", "activated_carbon", "steel_media", "grinding_media", "coagulant", "dispersant", "diesel", "hydrogen_peroxide", "ferric_sulphate"]
SUPPORTED_PRODUCT_TYPES = ["gold_dore", "copper_concentrate", "zinc_concentrate", "lead_concentrate", "nickel_concentrate", "iron_ore_lump", "iron_ore_fines", "coal_product", "silver_dore", "lithium_carbonate"]
SUPPORTED_SAMPLE_POINTS = ["feed", "cyclone_overflow", "cyclone_underflow", "flotation_feed", "concentrate", "tailings", "thickener_overflow", "thickener_underflow", "product", "reagent_addition_point"]
SUPPORTED_BALANCE_TYPES = ["daily", "weekly", "monthly", "campaign", "annual"]
SUPPORTED_RECOVERY_METHODS = ["assay_based", "mass_balance", "attributable_metal", "reconciliation"]
SUPPORTED_FEED_SOURCES = ["rom_ore", "crushed_ore", "stockpile_blend", "reclaimed", "purchased_ore", "reprocessed_tailings"]
SUPPORTED_QUALITY_PARAMETERS = ["grade", "moisture", "particle_size", "density", "recovery", "deleterious_elements", "mass_pull"]
SUPPORTED_CIRCUIT_STATUSES = ["running", "standby", "maintenance", "shutdown", "commissioning", "rampup"]
SUPPORTED_DEVIATION_TYPES = ["grade_deviation", "recovery_deviation", "throughput_deviation", "reagent_deviation", "quality_deviation"]
SUPPORTED_ALERT_LEVELS = ["critical", "high", "medium", "low"]
SUPPORTED_REVIEW_STATUSES = ["draft", "submitted", "approved", "rejected"]
SUPPORTED_RECONCILIATION_STATUSES = ["open", "submitted", "approved", "finalised"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"plant_feed": {
		"supported_feed_sources": SUPPORTED_FEED_SOURCES,
		"feed_grade_required": True,
		"feed_tonnage_required": True,
		"moisture_required": True,
	},
	"circuits": {
		"supported_circuits": SUPPORTED_PROCESS_CIRCUITS,
		"supported_statuses": SUPPORTED_CIRCUIT_STATUSES,
		"throughput_tracking_required": True,
	},
	"reagents": {
		"supported_types": SUPPORTED_REAGENT_TYPES,
		"dosage_rate_required": True,
		"inventory_tracking_required": True,
		"cyanide_code_compliance_required": True,
	},
	"metallurgical_balance": {
		"supported_types": SUPPORTED_BALANCE_TYPES,
		"supported_recovery_methods": SUPPORTED_RECOVERY_METHODS,
		"approval_required": True,
		"sign_off_required": True,
	},
	"product_quality": {
		"supported_product_types": SUPPORTED_PRODUCT_TYPES,
		"supported_quality_parameters": SUPPORTED_QUALITY_PARAMETERS,
		"specification_check_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"unapproved_balance_publication_denied": True,
		"cyanide_code_bypass_denied": True,
		"cross_tenant_read_denied": True,
		"off_spec_product_dispatch_denied": True,
	},
	"observability": {"event_stream": ORE_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "monitoring": "moni", "event_stream": "bytewax"},
	"ui": {
		"enable_dashboard": True,
		"enable_plant_feed": True,
		"enable_circuits": True,
		"enable_reagents": True,
		"enable_met_balance": True,
		"enable_product_quality": True,
		"enable_reconciliation": True,
	},
	"theme": {"default_theme": "mining_ore_process", "allow_tenant_overrides": True},
}

PROVIDES = [
	"plant_feed_tracking",
	"metallurgical_balance_workflow",
	"reagent_management",
	"recovery_optimisation_tracking",
	"product_quality_management",
	"process_circuit_monitoring",
	"ore_reconciliation_workflow",
	"deviation_alert_management",
	"assay_database_management",
	"process_kpi_reporting",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "moni", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mining-ore/dashboard", "component": "OreDashboard", "permission": "mining_ore:view", "nav_group": "Overview"},
	{"name": "plant_feed", "path": "/mining-ore/plant-feed", "component": "PlantFeedLedger", "permission": "mining_ore:view", "nav_group": "Plant Feed"},
	{"name": "plant_feed_record", "path": "/mining-ore/plant-feed/record", "component": "PlantFeedForm", "permission": "mining_ore:write", "nav_group": "Plant Feed"},
	{"name": "circuits", "path": "/mining-ore/circuits", "component": "CircuitStatusBoard", "permission": "mining_ore:view", "nav_group": "Process"},
	{"name": "circuit_detail", "path": "/mining-ore/circuits/:id", "component": "CircuitDetail", "permission": "mining_ore:view", "nav_group": "Process"},
	{"name": "reagents", "path": "/mining-ore/reagents", "component": "ReagentInventory", "permission": "mining_ore:view", "nav_group": "Reagents"},
	{"name": "reagent_usage", "path": "/mining-ore/reagents/usage", "component": "ReagentUsageLedger", "permission": "mining_ore:write", "nav_group": "Reagents"},
	{"name": "met_balance", "path": "/mining-ore/met-balance", "component": "MetallurgicalBalanceList", "permission": "mining_ore:met_balance", "nav_group": "Metallurgy"},
	{"name": "met_balance_detail", "path": "/mining-ore/met-balance/:id", "component": "MetBalanceDetail", "permission": "mining_ore:met_balance", "nav_group": "Metallurgy"},
	{"name": "product_quality", "path": "/mining-ore/product-quality", "component": "ProductQualityLedger", "permission": "mining_ore:view", "nav_group": "Quality"},
	{"name": "reconciliation", "path": "/mining-ore/reconciliation", "component": "ReconciliationConsole", "permission": "mining_ore:reconciliation", "nav_group": "Reconciliation"},
	{"name": "deviations", "path": "/mining-ore/deviations", "component": "DeviationAlertList", "permission": "mining_ore:view", "nav_group": "Alerts"},
	{"name": "reports", "path": "/mining-ore/reports", "component": "OreReportList", "permission": "mining_ore:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mining-ore/settings", "component": "OreSettings", "permission": "mining_ore:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mining_ore_process",
	"tokens": {
		"color.primary": "#065F46",
		"color.accent": "#7C3AED",
		"color.success": "#15803D",
		"color.warning": "#B45309",
		"color.danger": "#DC2626",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#064E3B",
		"text.secondary": "#065F46",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"plant_feed": {"icon": "package-open", "status_indicator": "feed-source-chip"},
		"circuits": {"icon": "git-branch", "status_indicator": "circuit-status-chip"},
		"reagents": {"icon": "flask", "status_indicator": "reagent-type-chip"},
		"met_balance": {"icon": "scale", "status_indicator": "balance-status-chip"},
		"product_quality": {"icon": "award", "status_indicator": "quality-spec-chip"},
		"reconciliation": {"icon": "check-circle-2", "status_indicator": "reconciliation-status-chip"},
		"deviations": {"icon": "alert-triangle", "status_indicator": "deviation-severity-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": ORE_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"plant_feed_recorded",
		"circuit_status_changed",
		"reagent_usage_recorded",
		"reagent_reorder_triggered",
		"metallurgical_balance_submitted",
		"metallurgical_balance_approved",
		"product_quality_recorded",
		"off_spec_product_flagged",
		"recovery_deviation_detected",
		"reconciliation_finalised",
		"grade_deviation_alert_raised",
	],
	"guardrails": [
		"unapproved_balance_publication_denied",
		"cyanide_code_bypass_denied",
		"off_spec_product_dispatch_denied",
		"cross_tenant_read_denied",
		"negative_recovery_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "feed_source_supported", "condition": {"operation": "record_plant_feed", "feed_source_supported": False}, "effect": {"decision": "deny", "reason": "feed_source_not_supported", "required_action": "select_supported_feed_source"}},
	{"name": "feed_grade_required", "condition": {"operation": "record_plant_feed", "feed_grade_present": False}, "effect": {"decision": "deny", "reason": "feed_grade_required", "required_action": "provide_feed_grade"}},
	{"name": "feed_tonnage_required", "condition": {"operation": "record_plant_feed", "feed_tonnage_present": False}, "effect": {"decision": "deny", "reason": "feed_tonnage_required", "required_action": "provide_feed_tonnage"}},
	{"name": "circuit_type_supported", "condition": {"operation": "update_circuit_status", "circuit_type_supported": False}, "effect": {"decision": "deny", "reason": "circuit_type_not_supported", "required_action": "select_supported_circuit_type"}},
	{"name": "reagent_type_supported", "condition": {"operation": "record_reagent_usage", "reagent_type_supported": False}, "effect": {"decision": "deny", "reason": "reagent_type_not_supported", "required_action": "select_supported_reagent_type"}},
	{"name": "reagent_dosage_required", "condition": {"operation": "record_reagent_usage", "dosage_rate_present": False}, "effect": {"decision": "deny", "reason": "dosage_rate_required", "required_action": "provide_dosage_rate"}},
	{"name": "cyanide_code_compliance", "condition": {"operation": "record_cyanide_usage", "cyanide_code_compliant": False}, "effect": {"decision": "deny", "reason": "icmc_cyanide_code_compliance_required", "required_action": "verify_cyanide_code_compliance"}},
	{"name": "met_balance_type_supported", "condition": {"operation": "submit_met_balance", "balance_type_supported": False}, "effect": {"decision": "deny", "reason": "balance_type_not_supported", "required_action": "select_supported_balance_type"}},
	{"name": "recovery_method_supported", "condition": {"operation": "submit_met_balance", "recovery_method_supported": False}, "effect": {"decision": "deny", "reason": "recovery_method_not_supported", "required_action": "select_supported_recovery_method"}},
	{"name": "met_balance_approval_required", "condition": {"operation": "publish_met_balance", "balance_approved": False}, "effect": {"decision": "deny", "reason": "metallurgical_balance_approval_required", "required_action": "obtain_balance_approval"}},
	{"name": "negative_recovery_denied", "condition": {"operation": "submit_met_balance", "recovery_negative": True}, "effect": {"decision": "deny", "reason": "negative_recovery_not_permitted", "required_action": "review_mass_balance_inputs"}},
	{"name": "recovery_over_100_denied", "condition": {"operation": "submit_met_balance", "recovery_over_100": True}, "effect": {"decision": "deny", "reason": "recovery_cannot_exceed_100_percent", "required_action": "review_mass_balance_inputs"}},
	{"name": "product_type_supported", "condition": {"operation": "record_product_quality", "product_type_supported": False}, "effect": {"decision": "deny", "reason": "product_type_not_supported", "required_action": "select_supported_product_type"}},
	{"name": "off_spec_dispatch_denied", "condition": {"operation": "dispatch_product", "product_meets_spec": False}, "effect": {"decision": "deny", "reason": "off_spec_product_cannot_be_dispatched", "required_action": "obtain_off_spec_dispatch_approval"}},
	{"name": "reconciliation_approval_required", "condition": {"operation": "finalise_reconciliation", "reconciliation_approved": False}, "effect": {"decision": "deny", "reason": "reconciliation_approval_required", "required_action": "obtain_reconciliation_approval"}},
	{"name": "cross_tenant_read_denied", "condition": {"operation": "read", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "delete_approved_balance_denied", "condition": {"operation": "delete", "balance_status": "approved"}, "effect": {"decision": "deny", "reason": "approved_balance_cannot_be_deleted", "required_action": "supersede_instead"}},
	{"name": "deviation_type_supported", "condition": {"operation": "raise_deviation_alert", "deviation_type_supported": False}, "effect": {"decision": "deny", "reason": "deviation_type_not_supported", "required_action": "select_supported_deviation_type"}},
	{"name": "sample_point_supported", "condition": {"operation": "record_sample", "sample_point_supported": False}, "effect": {"decision": "deny", "reason": "sample_point_not_supported", "required_action": "select_supported_sample_point"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	"""Return the full capability contract for the given tenant."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	return {
		"capability": CAPABILITY_ID,
		"display_name": CAPABILITY_NAME,
		"version": CAPABILITY_VERSION,
		"configuration": config,
		"configuration_schema": {
			"required": ["tenant_id", "ui", "theme"],
			"properties": {
				"tenant_id": {"type": "string"},
				"ui": {"type": "object"},
				"theme": {"type": "object"},
				"plant_feed": {"type": "object"},
				"reagents": {"type": "object"},
				"metallurgical_balance": {"type": "object"},
				"product_quality": {"type": "object"},
			},
		},
		"rule_engine": {
			"type": "deterministic",
			"default_decision": "allow",
			"rules": RULES,
		},
		"ui": {
			"shell": "apg_python",
			"requires_theme": True,
			"template_roots": ["mining/ore/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the given context dict."""
	matched_denials: list[dict[str, Any]] = []
	matched_allows: list[dict[str, Any]] = []

	for rule in RULES:
		condition = rule["condition"]
		all_match = all(context.get(k) == v for k, v in condition.items())
		if all_match:
			effect = rule["effect"]
			entry = {"rule": rule["name"], "effect": effect}
			if effect["decision"] == "deny":
				matched_denials.append(entry)
			else:
				matched_allows.append(entry)

	if matched_denials:
		return {
			"decision": "deny",
			"matched_denials": matched_denials,
			"matched_allows": matched_allows,
			"required_actions": [d["effect"]["required_action"] for d in matched_denials],
		}

	return {
		"decision": "allow",
		"matched_denials": [],
		"matched_allows": matched_allows,
		"required_actions": [],
	}
