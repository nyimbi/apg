"""Executable capability contract for APG Exploration Data Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "mining_exp"
CAPABILITY_NAME = "Exploration Data Management"
CAPABILITY_VERSION = "1.0.0"
EXP_EVENT_STREAM = "apg.mining.exp.lifecycle"

SUPPORTED_HOLE_TYPES = ["diamond", "rotary_air_blast", "reverse_circulation", "sonic", "auger", "percussion", "core"]
SUPPORTED_SAMPLE_TYPES = ["core", "chip", "channel", "grab", "soil", "stream_sediment", "rock_chip"]
SUPPORTED_LITHOLOGY_CODES = ["granite", "basalt", "sandstone", "limestone", "shale", "quartzite", "gneiss", "schist", "diorite", "rhyolite", "tuff", "conglomerate", "andesite", "dolerite", "peridotite"]
SUPPORTED_ASSAY_METHODS = ["fire_assay", "icp_ms", "icp_oes", "xrf", "aaas", "aqua_regia_digest", "four_acid_digest", "neutron_activation", "screen_fire"]
SUPPORTED_COMMODITIES = ["gold", "copper", "silver", "zinc", "lead", "nickel", "cobalt", "iron_ore", "coal", "uranium", "lithium", "manganese", "chromite", "platinum", "palladium"]
SUPPORTED_RESOURCE_CLASSIFICATIONS = ["measured", "indicated", "inferred", "exploration_target"]
SUPPORTED_RESERVE_CLASSIFICATIONS = ["proven", "probable"]
SUPPORTED_REPORTING_STANDARDS = ["jorc_2012", "ni_43_101", "samrec", "perc", "kazrc"]
SUPPORTED_COORDINATE_SYSTEMS = ["wgs84", "utm", "mga", "local_grid"]
SUPPORTED_SURVEY_TYPES = ["downhole_gyro", "mag_susceptibility", "resistivity", "ip", "em", "gravity", "radiometric", "seismic"]
SUPPORTED_QAQC_TYPES = ["blank", "standard", "duplicate_field", "duplicate_coarse", "duplicate_pulp", "check_assay"]
SUPPORTED_OXIDATION_STATES = ["fresh", "transitional", "oxidised", "supergene"]
SUPPORTED_MINERALISATION_STYLES = ["disseminated", "vein", "stockwork", "massive_sulphide", "breccia", "skarn", "porphyry", "epithermal", "orogenic"]
SUPPORTED_REVIEW_STATUSES = ["pending", "in_review", "approved", "rejected", "superseded"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"drill_holes": {
		"supported_hole_types": SUPPORTED_HOLE_TYPES,
		"collar_survey_required": True,
		"down_hole_survey_required": True,
		"coordinate_system_required": True,
	},
	"sampling": {
		"supported_sample_types": SUPPORTED_SAMPLE_TYPES,
		"from_to_required": True,
		"sample_id_unique": True,
		"qaqc_insertion_required": True,
	},
	"assays": {
		"supported_methods": SUPPORTED_ASSAY_METHODS,
		"lab_cert_required": True,
		"detection_limit_required": True,
		"qaqc_review_required": True,
	},
	"geology": {
		"supported_lithology_codes": SUPPORTED_LITHOLOGY_CODES,
		"supported_oxidation_states": SUPPORTED_OXIDATION_STATES,
		"supported_mineralisation_styles": SUPPORTED_MINERALISATION_STYLES,
	},
	"resources": {
		"supported_classifications": SUPPORTED_RESOURCE_CLASSIFICATIONS,
		"supported_reserve_classifications": SUPPORTED_RESERVE_CLASSIFICATIONS,
		"supported_reporting_standards": SUPPORTED_REPORTING_STANDARDS,
		"competent_person_required": True,
		"estimation_method_required": True,
	},
	"reporting": {
		"supported_standards": SUPPORTED_REPORTING_STANDARDS,
		"public_disclosure_review_required": True,
		"competent_person_sign_off_required": True,
	},
	"governance": {
		"require_tenant_context": True,
		"policy_attached_for_writes": True,
		"audit_events": True,
		"cross_tenant_read_denied": True,
		"unapproved_resource_publication_denied": True,
		"qaqc_bypass_denied": True,
	},
	"observability": {"event_stream": EXP_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "nlp": "nlpc", "geospatial": "geos", "search": "srch", "workflow": "wflo", "event_stream": "bytewax"},
	"ui": {
		"enable_dashboard": True,
		"enable_drillholes": True,
		"enable_assays": True,
		"enable_geology": True,
		"enable_resources": True,
		"enable_qaqc": True,
		"enable_maps": True,
		"enable_reports": True,
	},
	"theme": {"default_theme": "mining_exp_geo", "allow_tenant_overrides": True},
}

PROVIDES = [
	"drillhole_collar_management",
	"downhole_survey_management",
	"lithology_logging",
	"assay_data_management",
	"qaqc_monitoring",
	"resource_estimation_workflow",
	"jorc_reporting_workflow",
	"ni_43_101_reporting_workflow",
	"geological_map_management",
	"exploration_target_delineation",
]

REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "nlpc", "geos", "srch", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/mining-exp/dashboard", "component": "ExpDashboard", "permission": "mining_exp:view", "nav_group": "Overview"},
	{"name": "drillholes", "path": "/mining-exp/drillholes", "component": "DrillholeList", "permission": "mining_exp:view", "nav_group": "Field Data"},
	{"name": "drillhole_create", "path": "/mining-exp/drillholes/create", "component": "DrillholeForm", "permission": "mining_exp:write", "nav_group": "Field Data"},
	{"name": "drillhole_detail", "path": "/mining-exp/drillholes/:id", "component": "DrillholeDetail", "permission": "mining_exp:view", "nav_group": "Field Data"},
	{"name": "assays", "path": "/mining-exp/assays", "component": "AssayLedger", "permission": "mining_exp:view", "nav_group": "Geochemistry"},
	{"name": "assay_import", "path": "/mining-exp/assays/import", "component": "AssayImport", "permission": "mining_exp:write", "nav_group": "Geochemistry"},
	{"name": "geology", "path": "/mining-exp/geology", "component": "GeologyLog", "permission": "mining_exp:view", "nav_group": "Geology"},
	{"name": "qaqc", "path": "/mining-exp/qaqc", "component": "QAQCDashboard", "permission": "mining_exp:view", "nav_group": "Quality"},
	{"name": "resources", "path": "/mining-exp/resources", "component": "ResourceEstimateList", "permission": "mining_exp:resources", "nav_group": "Resources"},
	{"name": "resource_detail", "path": "/mining-exp/resources/:id", "component": "ResourceEstimateDetail", "permission": "mining_exp:resources", "nav_group": "Resources"},
	{"name": "maps", "path": "/mining-exp/maps", "component": "GeologicalMapViewer", "permission": "mining_exp:view", "nav_group": "Spatial"},
	{"name": "reports", "path": "/mining-exp/reports", "component": "ComplianceReportList", "permission": "mining_exp:reports", "nav_group": "Reporting"},
	{"name": "report_create", "path": "/mining-exp/reports/create", "component": "ComplianceReportForm", "permission": "mining_exp:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/mining-exp/settings", "component": "ExpSettings", "permission": "mining_exp:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "mining_exp_geo",
	"tokens": {
		"color.primary": "#78350F",
		"color.accent": "#15803D",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#FAFAF9",
		"surface.panel": "#FFFFFF",
		"text.primary": "#1C1917",
		"text.secondary": "#57534E",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"drillholes": {"icon": "drill", "status_indicator": "hole-type-chip"},
		"assays": {"icon": "flask-conical", "status_indicator": "qaqc-status-chip"},
		"geology": {"icon": "layers", "status_indicator": "lithology-chip"},
		"qaqc": {"icon": "shield-check", "status_indicator": "qaqc-flag-chip"},
		"resources": {"icon": "gem", "status_indicator": "classification-chip"},
		"maps": {"icon": "map", "status_indicator": "map-type-chip"},
		"reports": {"icon": "file-text", "status_indicator": "report-status-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": EXP_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"drillhole_collar_recorded",
		"downhole_survey_recorded",
		"lithology_interval_logged",
		"assay_result_imported",
		"qaqc_flag_raised",
		"qaqc_flag_resolved",
		"resource_estimate_submitted",
		"resource_estimate_approved",
		"compliance_report_published",
		"geological_map_updated",
	],
	"guardrails": [
		"unapproved_resource_publication_denied",
		"qaqc_bypass_denied",
		"cross_tenant_read_denied",
		"competent_person_required_for_resource",
		"assay_without_collar_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "write_policy_required", "required_action": "attach_write_policy"}},
	{"name": "hole_type_supported", "condition": {"operation": "create_drillhole", "hole_type_supported": False}, "effect": {"decision": "deny", "reason": "hole_type_not_supported", "required_action": "select_supported_hole_type"}},
	{"name": "collar_coordinates_required", "condition": {"operation": "create_drillhole", "collar_coordinates_present": False}, "effect": {"decision": "deny", "reason": "collar_coordinates_required", "required_action": "provide_collar_coordinates"}},
	{"name": "collar_coordinate_system_required", "condition": {"operation": "create_drillhole", "coordinate_system_present": False}, "effect": {"decision": "deny", "reason": "coordinate_system_required", "required_action": "specify_coordinate_system"}},
	{"name": "drillhole_id_unique", "condition": {"operation": "create_drillhole", "hole_id_unique": False}, "effect": {"decision": "deny", "reason": "drillhole_id_must_be_unique", "required_action": "provide_unique_hole_id"}},
	{"name": "assay_requires_collar", "condition": {"operation": "import_assays", "collar_exists": False}, "effect": {"decision": "deny", "reason": "assay_requires_existing_collar", "required_action": "create_collar_first"}},
	{"name": "assay_from_to_required", "condition": {"operation": "import_assays", "from_to_present": False}, "effect": {"decision": "deny", "reason": "from_to_intervals_required", "required_action": "provide_from_to_intervals"}},
	{"name": "assay_method_supported", "condition": {"operation": "import_assays", "assay_method_supported": False}, "effect": {"decision": "deny", "reason": "assay_method_not_supported", "required_action": "select_supported_assay_method"}},
	{"name": "assay_lab_cert_required", "condition": {"operation": "import_assays", "lab_cert_present": False}, "effect": {"decision": "deny", "reason": "lab_certificate_required", "required_action": "attach_lab_certificate"}},
	{"name": "qaqc_insertion_required", "condition": {"operation": "submit_sample_batch", "qaqc_inserted": False}, "effect": {"decision": "deny", "reason": "qaqc_samples_required", "required_action": "insert_qaqc_samples"}},
	{"name": "qaqc_bypass_denied", "condition": {"operation": "bypass_qaqc_check", "has_override_authority": False}, "effect": {"decision": "deny", "reason": "qaqc_bypass_not_permitted", "required_action": "complete_qaqc_review"}},
	{"name": "geology_from_to_required", "condition": {"operation": "log_geology", "from_to_present": False}, "effect": {"decision": "deny", "reason": "from_to_intervals_required", "required_action": "provide_from_to_intervals"}},
	{"name": "lithology_code_supported", "condition": {"operation": "log_geology", "lithology_code_supported": False}, "effect": {"decision": "deny", "reason": "lithology_code_not_supported", "required_action": "select_supported_lithology_code"}},
	{"name": "resource_competent_person_required", "condition": {"operation": "submit_resource_estimate", "competent_person_present": False}, "effect": {"decision": "deny", "reason": "competent_person_required", "required_action": "assign_competent_person"}},
	{"name": "resource_classification_supported", "condition": {"operation": "submit_resource_estimate", "classification_supported": False}, "effect": {"decision": "deny", "reason": "classification_not_supported", "required_action": "select_supported_classification"}},
	{"name": "resource_estimation_method_required", "condition": {"operation": "submit_resource_estimate", "estimation_method_present": False}, "effect": {"decision": "deny", "reason": "estimation_method_required", "required_action": "specify_estimation_method"}},
	{"name": "resource_approval_required_for_publication", "condition": {"operation": "publish_resource_estimate", "approved": False}, "effect": {"decision": "deny", "reason": "approval_required_before_publication", "required_action": "obtain_approval"}},
	{"name": "reporting_standard_supported", "condition": {"operation": "create_compliance_report", "reporting_standard_supported": False}, "effect": {"decision": "deny", "reason": "reporting_standard_not_supported", "required_action": "select_supported_reporting_standard"}},
	{"name": "report_competent_person_sign_off", "condition": {"operation": "publish_compliance_report", "competent_person_signed": False}, "effect": {"decision": "deny", "reason": "competent_person_sign_off_required", "required_action": "obtain_competent_person_signature"}},
	{"name": "cross_tenant_read_denied", "condition": {"operation": "read", "cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_own_tenant_context"}},
	{"name": "delete_approved_resource_denied", "condition": {"operation": "delete", "resource_status": "approved"}, "effect": {"decision": "deny", "reason": "approved_resource_cannot_be_deleted", "required_action": "supersede_instead"}},
	{"name": "interval_overlap_check", "condition": {"operation": "import_assays", "interval_overlap_detected": True}, "effect": {"decision": "deny", "reason": "overlapping_intervals_not_permitted", "required_action": "resolve_interval_overlap"}},
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
				"drill_holes": {"type": "object"},
				"sampling": {"type": "object"},
				"assays": {"type": "object"},
				"resources": {"type": "object"},
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
			"template_roots": ["mining/exp/templates"],
			"routes": UI_ROUTES,
		},
		"theme": THEME,
		"provides": PROVIDES,
		"requires": REQUIRES,
		"streaming": STREAMING,
	}


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate deterministic rules against the given context dict.

	Returns a result with decision, matched rules, and required actions.
	"""
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
