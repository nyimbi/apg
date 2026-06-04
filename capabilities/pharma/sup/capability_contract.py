"""Executable capability contract for APG Pharma Supply Chain."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pharma_sup"
CAPABILITY_NAME = "Pharmaceutical Supply Chain"
CAPABILITY_VERSION = "1.0.0"
SUP_EVENT_STREAM = "apg.pharma.sup.lifecycle"

SUPPORTED_SUPPLIER_TYPES = ["api_manufacturer", "excipient_supplier", "packaging_supplier", "cmo", "cdmo", "3pl", "broker", "trading_company", "distributor"]
SUPPORTED_QUALIFICATION_STATUSES = ["unqualified", "under_qualification", "qualified", "conditionally_qualified", "suspended", "disqualified", "re_qualification_required"]
SUPPORTED_CMO_TYPES = ["drug_substance", "drug_product", "fill_finish", "packaging", "testing_laboratory", "formulation_development", "clinical_manufacturing"]
SUPPORTED_DEMAND_METHODS = ["statistical_forecast", "consensus_forecast", "causal_forecast", "collaborative_planning", "s_op", "top_down", "bottom_up"]
SUPPORTED_IMPORT_LICENSE_TYPES = ["import_permit", "import_license", "narcotics_license", "controlled_substances", "investigational_product", "emergency_use", "humanitarian"]
SUPPORTED_SECURITY_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_SUPPLY_STATUSES = ["secure", "at_risk", "shortage", "out_of_stock", "discontinued", "allocated", "constrained"]
SUPPORTED_ORDER_TYPES = ["purchase_order", "forecast_order", "blanket_order", "emergency_order", "sample_order", "clinical_supply_order", "return_order"]
SUPPORTED_TRANSPORT_CONDITIONS = ["ambient", "cold_chain_2_8", "frozen", "controlled_room_temp", "dry_ice", "liquid_nitrogen"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["supplier_qualifier", "demand_planner", "cmo_manager", "import_coordinator", "supply_risk_monitor"]
SUPPORTED_REGULATORY_REGIONS = ["us_fda", "eu_ema", "uk_mhra", "japan_pmda", "canada_health", "australia_tga", "brazil_anvisa", "india_cdsco", "china_nmpa"]
SUPPORTED_CONTRACT_TYPES = ["quality_agreement", "supply_agreement", "confidentiality", "service_level", "technical_agreement", "manufacturing_agreement"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"suppliers": {"supported_types": SUPPORTED_SUPPLIER_TYPES, "supported_qualification_statuses": SUPPORTED_QUALIFICATION_STATUSES, "quality_agreement_required": True, "audit_cycle_months": 24, "approved_supplier_list_required": True},
	"cmo": {"supported_types": SUPPORTED_CMO_TYPES, "technical_agreement_required": True, "quality_agreement_required": True, "manufacturing_agreement_required": True, "site_audit_required": True},
	"demand_planning": {"supported_methods": SUPPORTED_DEMAND_METHODS, "demand_review_frequency": "monthly", "sop_review_frequency": "quarterly", "forecast_horizon_months": 24, "safety_stock_calculation": True},
	"import_licensing": {"supported_types": SUPPORTED_IMPORT_LICENSE_TYPES, "supported_regions": SUPPORTED_REGULATORY_REGIONS, "renewal_alert_days": 90, "expiry_tracking_required": True, "authority_reference_required": True},
	"supply_security": {"supported_risk_levels": SUPPORTED_SECURITY_RISK_LEVELS, "supported_supply_statuses": SUPPORTED_SUPPLY_STATUSES, "dual_sourcing_threshold": "high", "shortage_reporting_required": True, "contingency_plan_required": True},
	"orders": {"supported_types": SUPPORTED_ORDER_TYPES, "supported_transport_conditions": SUPPORTED_TRANSPORT_CONDITIONS, "quality_release_required": True, "coa_required": True, "serialisation_check_required": True},
	"contracts": {"supported_types": SUPPORTED_CONTRACT_TYPES, "version_control_required": True, "approval_required": True, "expiry_tracking_required": True, "renewal_alert_days": 60},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "approved_supplier_list_enforced": True, "quality_agreement_required": True, "cross_tenant_denied": True},
	"observability": {"event_stream": SUP_EVENT_STREAM, "stream_processor": "bytewax"},
	"adapters": {"auth": "auth", "audit": "audl", "notifications": "ntfy", "workflow": "wflo", "compliance": "comp", "monitoring": "moni", "scheduler": "schd", "event_stream": "mqeb"},
	"ui": {"enable_dashboard": True, "enable_suppliers": True, "enable_cmo": True, "enable_demand_planning": True, "enable_import_licensing": True, "enable_supply_security": True, "enable_orders": True, "enable_contracts": True},
	"theme": {"default_theme": "pharma_sup_chain", "allow_tenant_overrides": True},
}

PROVIDES = [
	"active_ingredient_sourcing_workflow",
	"cmo_management_workflow",
	"demand_planning_workflow",
	"import_licensing_workflow",
	"supply_security_monitoring_workflow",
	"supplier_qualification_workflow",
	"purchase_order_workflow",
	"supply_contract_workflow",
	"approved_supplier_list_workflow",
	"supply_risk_workflow",
]
REQUIRES = ["auth", "audl", "mten", "conf", "ntfy", "wflo", "comp", "moni", "schd", "mqeb"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/pharma-sup/dashboard", "component": "SupDashboard", "permission": "pharma_sup:view", "nav_group": "Overview"},
	{"name": "suppliers", "path": "/pharma-sup/suppliers", "component": "SupplierRegistry", "permission": "pharma_sup:suppliers", "nav_group": "Suppliers"},
	{"name": "supplier_detail", "path": "/pharma-sup/suppliers/<id>", "component": "SupplierDetail", "permission": "pharma_sup:suppliers", "nav_group": "Suppliers"},
	{"name": "approved_supplier_list", "path": "/pharma-sup/asl", "component": "ApprovedSupplierList", "permission": "pharma_sup:asl", "nav_group": "Suppliers"},
	{"name": "cmo", "path": "/pharma-sup/cmo", "component": "CmoManagement", "permission": "pharma_sup:cmo", "nav_group": "CMO"},
	{"name": "cmo_detail", "path": "/pharma-sup/cmo/<id>", "component": "CmoDetail", "permission": "pharma_sup:cmo", "nav_group": "CMO"},
	{"name": "demand_planning", "path": "/pharma-sup/demand", "component": "DemandPlanning", "permission": "pharma_sup:demand", "nav_group": "Planning"},
	{"name": "sop", "path": "/pharma-sup/sop", "component": "SopReview", "permission": "pharma_sup:sop", "nav_group": "Planning"},
	{"name": "orders", "path": "/pharma-sup/orders", "component": "OrderManagement", "permission": "pharma_sup:orders", "nav_group": "Operations"},
	{"name": "import_licensing", "path": "/pharma-sup/import-licenses", "component": "ImportLicenseRegistry", "permission": "pharma_sup:import", "nav_group": "Licensing"},
	{"name": "supply_security", "path": "/pharma-sup/security", "component": "SupplySecurityMonitor", "permission": "pharma_sup:security", "nav_group": "Risk"},
	{"name": "contracts", "path": "/pharma-sup/contracts", "component": "ContractVault", "permission": "pharma_sup:contracts", "nav_group": "Contracts"},
	{"name": "reports", "path": "/pharma-sup/reports", "component": "SupplyChainReports", "permission": "pharma_sup:reports", "nav_group": "Reporting"},
	{"name": "settings", "path": "/pharma-sup/settings", "component": "SupSettings", "permission": "pharma_sup:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "pharma_sup_chain",
	"tokens": {
		"color.primary": "#065F46",
		"color.accent": "#0369A1",
		"color.success": "#15803D",
		"color.warning": "#92400E",
		"color.danger": "#B91C1C",
		"surface.canvas": "#ECFDF5",
		"surface.panel": "#FFFFFF",
		"text.primary": "#064E3B",
		"text.secondary": "#374151",
		"border.radius": "6px",
		"density": "comfortable",
	},
	"components": {
		"suppliers": {"icon": "factory", "status_indicator": "supplier-qualification-chip"},
		"cmo": {"icon": "beaker", "status_indicator": "cmo-type-chip"},
		"demand_planning": {"icon": "bar-chart-2", "status_indicator": "demand-method-chip"},
		"import_licensing": {"icon": "file-check", "status_indicator": "import-license-type-chip"},
		"supply_security": {"icon": "shield-alert", "status_indicator": "supply-status-chip"},
		"orders": {"icon": "shopping-cart", "status_indicator": "order-type-chip"},
		"contracts": {"icon": "file-signature", "status_indicator": "contract-type-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": SUP_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"supplier_qualified", "supplier_suspended", "supplier_audit_completed",
		"cmo_activated", "cmo_agreement_signed",
		"demand_forecast_updated", "sop_completed",
		"order_placed", "order_received",
		"import_license_granted", "import_license_expiring", "import_license_expired",
		"supply_shortage_detected", "supply_risk_escalated",
		"contract_approved", "contract_expiring",
	],
	"guardrails": [
		"approved_supplier_list_enforced",
		"quality_agreement_required_before_supply",
		"import_license_required_for_imports",
		"shortage_reporting_required",
		"dual_sourcing_required_for_high_risk",
		"cmo_technical_agreement_required",
		"cross_tenant_supply_data_denied",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "policy_required", "required_action": "attach_policy"}},
	{"name": "supplier_type_supported", "condition": {"operation": "create_supplier", "supplier_type_supported": False}, "effect": {"decision": "deny", "reason": "supplier_type_not_supported", "required_action": "select_supported_supplier_type"}},
	{"name": "approved_supplier_list_required", "condition": {"operation": "place_order", "supplier_on_asl": False}, "effect": {"decision": "deny", "reason": "supplier_not_on_approved_list", "required_action": "qualify_supplier_for_asl"}},
	{"name": "quality_agreement_required", "condition": {"operation": "activate_supplier", "quality_agreement_signed": False}, "effect": {"decision": "deny", "reason": "quality_agreement_required", "required_action": "sign_quality_agreement"}},
	{"name": "supplier_qualification_required", "condition": {"operation": "place_order", "supplier_qualified": False}, "effect": {"decision": "deny", "reason": "supplier_qualification_required", "required_action": "complete_supplier_qualification"}},
	{"name": "cmo_type_supported", "condition": {"operation": "activate_cmo", "cmo_type_supported": False}, "effect": {"decision": "deny", "reason": "cmo_type_not_supported", "required_action": "select_supported_cmo_type"}},
	{"name": "cmo_technical_agreement_required", "condition": {"operation": "activate_cmo", "technical_agreement_signed": False}, "effect": {"decision": "deny", "reason": "technical_agreement_required", "required_action": "sign_technical_agreement"}},
	{"name": "cmo_quality_agreement_required", "condition": {"operation": "activate_cmo", "quality_agreement_signed": False}, "effect": {"decision": "deny", "reason": "cmo_quality_agreement_required", "required_action": "sign_cmo_quality_agreement"}},
	{"name": "demand_method_supported", "condition": {"operation": "create_forecast", "demand_method_supported": False}, "effect": {"decision": "deny", "reason": "demand_method_not_supported", "required_action": "select_supported_demand_method"}},
	{"name": "import_license_required", "condition": {"operation": "import_shipment", "import_license_active": False}, "effect": {"decision": "deny", "reason": "import_license_required", "required_action": "obtain_import_license"}},
	{"name": "import_license_type_supported", "condition": {"operation": "apply_import_license", "license_type_supported": False}, "effect": {"decision": "deny", "reason": "import_license_type_not_supported", "required_action": "select_supported_license_type"}},
	{"name": "import_license_renewal_90d", "condition": {"operation": "check_import_license", "expiring_within_90d": True, "renewal_initiated": False}, "effect": {"decision": "deny", "reason": "import_license_renewal_required", "required_action": "initiate_import_license_renewal"}},
	{"name": "supply_shortage_reporting_required", "condition": {"operation": "update_supply_status", "status": "shortage", "regulatory_notified": False}, "effect": {"decision": "deny", "reason": "shortage_reporting_required", "required_action": "notify_regulatory_authority"}},
	{"name": "order_type_supported", "condition": {"operation": "place_order", "order_type_supported": False}, "effect": {"decision": "deny", "reason": "order_type_not_supported", "required_action": "select_supported_order_type"}},
	{"name": "order_coa_required", "condition": {"operation": "receive_order", "coa_present": False}, "effect": {"decision": "deny", "reason": "coa_required_on_receipt", "required_action": "obtain_coa"}},
	{"name": "contract_approval_required", "condition": {"operation": "activate_contract", "approved": False}, "effect": {"decision": "deny", "reason": "contract_approval_required", "required_action": "obtain_contract_approval"}},
	{"name": "contract_expiry_renewal_60d", "condition": {"operation": "check_contract", "expiring_within_60d": True, "renewal_initiated": False}, "effect": {"decision": "deny", "reason": "contract_renewal_required", "required_action": "initiate_contract_renewal"}},
	{"name": "high_risk_dual_sourcing_required", "condition": {"operation": "confirm_supply_plan", "risk_level": "high", "dual_sourced": False}, "effect": {"decision": "deny", "reason": "dual_sourcing_required_for_high_risk", "required_action": "identify_alternate_supplier"}},
	{"name": "cross_tenant_denied", "condition": {"operation_type": "write", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_operation_denied", "required_action": "use_correct_tenant_context"}},
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
			"api_prefix": "/pharma-sup/api/v1",
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
