"""Executable APG capability contract for Product Information Management."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "pde_pim"
CAPABILITY_NAME = "Product Information Management"
CAPABILITY_VERSION = "2.1.0"
PIM_EVENT_STREAM = "apg.pde.pim.lifecycle"

SUPPORTED_PRODUCT_TYPES = ["physical", "digital", "service", "bundle", "component", "raw_material"]
SUPPORTED_LIFECYCLE_STAGES = ["concept", "design", "prototype", "active", "retired", "archived"]
SUPPORTED_ATTRIBUTE_TYPES = ["text", "number", "boolean", "date", "enum", "money", "media", "rich_text"]
SUPPORTED_CHANNELS = ["web", "marketplace", "erp", "pos", "print", "api"]
SUPPORTED_QUALITY_STATUSES = ["draft", "review", "approved", "published"]
SUPPORTED_RISK_TIERS = ["low", "medium", "high", "critical"]
SUPPORTED_PIM_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_PIM_AGENT_ROLES = ["catalog_reviewer", "data_quality_reviewer", "enrichment_reviewer", "channel_reviewer", "compliance_reviewer", "product_query_reviewer"]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"catalogs": {"code_required": True, "name_required": True, "owner_required": True},
	"products": {"sku_required": True, "name_required": True, "supported_types": SUPPORTED_PRODUCT_TYPES, "catalog_required": True, "owner_required": True},
	"attributes": {"code_required": True, "name_required": True, "supported_types": SUPPORTED_ATTRIBUTE_TYPES, "owner_required": True},
	"attribute_values": {"product_required": True, "attribute_required": True, "value_required": True, "locale_required_for_rich_text": True},
	"variants": {"parent_required": True, "sku_required": True, "option_values_required": True},
	"content": {"product_required": True, "locale_required": True, "title_required": True, "review_required_for_generated_content": True},
	"assets": {"product_required": True, "url_required": True, "asset_type_required": True, "rights_basis_required": True},
	"compliance": {"product_required": True, "framework_required": True, "status_required": True, "evidence_required": True, "review_required_for_high_risk": True},
	"channels": {"product_required": True, "supported_channels": SUPPORTED_CHANNELS, "listing_id_required": True, "approval_required": True},
	"publishing": {"approved_content_required": True, "approved_channel_required": True, "approval_required": True},
	"quality": {"severity_owner_required": ["high", "critical"], "supported_statuses": SUPPORTED_QUALITY_STATUSES},
	"changes": {"product_required": True, "reason_required": True, "approval_required": True},
	"pim_agents": {"enabled": True, "supported_runtimes": SUPPORTED_PIM_AGENT_RUNTIMES, "supported_roles": SUPPORTED_PIM_AGENT_ROLES, "max_autonomous_scope": "inspect_prepare_and_recommend", "human_approval_required": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_state_changes": True},
	"observability": {"event_stream": PIM_EVENT_STREAM, "stream_processor": "bytewax", "emit_product_events": True, "emit_content_events": True, "emit_publish_events": True, "emit_agent_events": True},
	"adapters": {"authorization": "adapter", "audit": "adapter", "notification": "adapter", "workflow": "adapter", "media": "adapter", "commerce": "adapter", "erp": "adapter", "event_stream": "bytewax", "theme": "adapter"},
	"ui": {"enable_dashboard": True, "enable_catalogs": True, "enable_products": True, "enable_attributes": True, "enable_content": True, "enable_assets": True, "enable_compliance": True, "enable_channels": True, "enable_quality": True, "enable_changes": True, "enable_agents": True, "enable_settings": True},
	"theme": {"default_theme": "pim_control", "allow_tenant_overrides": True},
}


PROVIDES = ["product_catalog_lifecycle", "product_record_lifecycle", "product_attribute_lifecycle", "product_variant_lifecycle", "product_content_lifecycle", "product_asset_lifecycle", "product_compliance_lifecycle", "product_channel_listing_lifecycle", "product_publish_workflow", "product_data_quality_workflow", "product_change_workflow", "pim_dashboard_service", "pim_agents"]
REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_events",
	"composition_config",
	"wflo",
	"mdm",
	"onto",
]


UI_ROUTES = [
	{"name": "dashboard", "path": "/pde/pim/dashboard", "component": "PIMDashboard", "permission": "pde_pim:view", "nav_group": "Overview"},
	{"name": "catalogs", "path": "/pde/pim/catalogs", "component": "CatalogWorkbench", "permission": "pde_pim:manage_catalogs", "nav_group": "Setup"},
	{"name": "products", "path": "/pde/pim/products", "component": "ProductWorkbench", "permission": "pde_pim:manage_products", "nav_group": "Products"},
	{"name": "attributes", "path": "/pde/pim/attributes", "component": "AttributeCatalog", "permission": "pde_pim:manage_attributes", "nav_group": "Setup"},
	{"name": "content", "path": "/pde/pim/content", "component": "ProductContentDesk", "permission": "pde_pim:manage_content", "nav_group": "Content"},
	{"name": "assets", "path": "/pde/pim/assets", "component": "ProductAssetDesk", "permission": "pde_pim:manage_assets", "nav_group": "Content"},
	{"name": "compliance", "path": "/pde/pim/compliance", "component": "ProductComplianceDesk", "permission": "pde_pim:govern", "nav_group": "Governance"},
	{"name": "channels", "path": "/pde/pim/channels", "component": "ChannelListingDesk", "permission": "pde_pim:publish", "nav_group": "Publishing"},
	{"name": "quality", "path": "/pde/pim/quality", "component": "DataQualityCenter", "permission": "pde_pim:quality", "nav_group": "Governance"},
	{"name": "changes", "path": "/pde/pim/changes", "component": "ProductChangeDesk", "permission": "pde_pim:approve_changes", "nav_group": "Governance"},
	{"name": "agents", "path": "/pde/pim/agents", "component": "PIMAgentWorkbench", "permission": "pde_pim:agent_manage", "nav_group": "Automation"},
	{"name": "rules", "path": "/pde/pim/rules", "component": "PIMRules", "permission": "pde_pim:govern", "nav_group": "Governance"},
	{"name": "settings", "path": "/pde/pim/settings", "component": "PIMSettings", "permission": "pde_pim:admin", "nav_group": "Administration"},
]


THEME = {"name": "pim_control", "tokens": {"border.radius": "8px", "color.primary": "#28536B", "color.accent": "#8A5A20", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "density": "compact"}, "components": {"dashboard": {"icon": "layout-dashboard", "status_indicator": "health-pill", "visual": "catalog-grid"}, "products": {"icon": "package", "status_style": "product-chip", "visual": "product-table"}, "content": {"icon": "file-text", "status_style": "content-chip", "visual": "content-workbench"}, "channels": {"icon": "send", "status_style": "channel-chip", "visual": "listing-board"}, "quality": {"icon": "badge-check", "status_style": "quality-chip", "visual": "issue-list"}, "agents": {"icon": "bot", "status_style": "agent-chip", "visual": "agent-roster"}, "rules": {"icon": "list-checks", "status_style": "decision-chip", "visual": "rule-list"}, "settings": {"icon": "settings", "density": "compact", "visual": "settings-panel"}}}


STREAMING = {"processor": "bytewax", "event_stream": PIM_EVENT_STREAM, "events": ["catalog_created", "product_created", "attribute_defined", "attribute_value_set", "variant_created", "content_enriched", "asset_attached", "compliance_recorded", "channel_listing_created", "product_published", "quality_issue_recorded", "change_request_created", "change_request_approved", "pim_agent_registered"], "delivery": "at_least_once", "ordering_key": "tenant_id"}


def _rule(name: str, description: str, condition: dict[str, Any], decision: str, reason: str, action: str) -> dict[str, Any]:
	return {"name": name, "description": description, "condition": condition, "effect": {"decision": decision, "reason": reason, "required_action": action}}


RULES = [
	_rule("tenant_context_required", "PIM operations require tenant context.", {"tenant_context_present": False}, "deny", "tenant_context_required", "attach_tenant_context"),
	_rule("operation_policy_required", "PIM write operations require policy enforcement.", {"operation_type": "write", "policy_attached": False}, "deny", "operation_policy_required", "attach_operation_policy"),
	_rule("catalog_code_required", "Catalogs require a code.", {"operation": "create_catalog", "code_present": False}, "deny", "catalog_code_required", "provide_catalog_code"),
	_rule("catalog_name_required", "Catalogs require a name.", {"operation": "create_catalog", "name_present": False}, "deny", "catalog_name_required", "provide_catalog_name"),
	_rule("catalog_owner_required", "Catalogs require an owner.", {"operation": "create_catalog", "owner_present": False}, "deny", "catalog_owner_required", "assign_owner"),
	_rule("product_catalog_required", "Products require a catalog.", {"operation": "create_product", "catalog_present": False}, "deny", "product_catalog_required", "select_catalog"),
	_rule("product_sku_required", "Products require a SKU.", {"operation": "create_product", "sku_present": False}, "deny", "product_sku_required", "provide_sku"),
	_rule("product_name_required", "Products require a name.", {"operation": "create_product", "name_present": False}, "deny", "product_name_required", "provide_name"),
	_rule("product_type_supported", "Product type must be supported.", {"operation": "create_product", "product_type_supported": False}, "deny", "product_type_not_supported", "choose_supported_product_type"),
	_rule("product_owner_required", "Products require an owner.", {"operation": "create_product", "owner_present": False}, "deny", "product_owner_required", "assign_owner"),
	_rule("attribute_code_required", "Attributes require a code.", {"operation": "define_attribute", "code_present": False}, "deny", "attribute_code_required", "provide_attribute_code"),
	_rule("attribute_type_supported", "Attribute type must be supported.", {"operation": "define_attribute", "attribute_type_supported": False}, "deny", "attribute_type_not_supported", "choose_supported_attribute_type"),
	_rule("attribute_owner_required", "Attributes require an owner.", {"operation": "define_attribute", "owner_present": False}, "deny", "attribute_owner_required", "assign_owner"),
	_rule("attribute_value_product_required", "Attribute values require a product.", {"operation": "set_attribute_value", "product_present": False}, "deny", "attribute_value_product_required", "select_product"),
	_rule("attribute_value_attribute_required", "Attribute values require an attribute.", {"operation": "set_attribute_value", "attribute_present": False}, "deny", "attribute_value_attribute_required", "select_attribute"),
	_rule("attribute_value_required", "Attribute values require a value.", {"operation": "set_attribute_value", "value_present": False}, "deny", "attribute_value_required", "provide_value"),
	_rule("rich_text_locale_required", "Rich text values require locale.", {"operation": "set_attribute_value", "rich_text": True, "locale_present": False}, "deny", "attribute_locale_required", "provide_locale"),
	_rule("variant_parent_required", "Variants require a parent product.", {"operation": "create_variant", "parent_present": False}, "deny", "variant_parent_required", "select_parent_product"),
	_rule("variant_sku_required", "Variants require a SKU.", {"operation": "create_variant", "sku_present": False}, "deny", "variant_sku_required", "provide_sku"),
	_rule("variant_options_required", "Variants require option values.", {"operation": "create_variant", "options_present": False}, "deny", "variant_options_required", "provide_options"),
	_rule("content_product_required", "Content requires a product.", {"operation": "enrich_content", "product_present": False}, "deny", "content_product_required", "select_product"),
	_rule("content_locale_required", "Content requires locale.", {"operation": "enrich_content", "locale_present": False}, "deny", "content_locale_required", "provide_locale"),
	_rule("content_title_required", "Content requires title.", {"operation": "enrich_content", "title_present": False}, "deny", "content_title_required", "provide_title"),
	_rule("generated_content_review_required", "Generated product content requires review.", {"operation": "enrich_content", "generated_content": True, "review_recorded": False}, "require_review", "generated_content_review_required", "record_content_review"),
	_rule("asset_product_required", "Assets require a product.", {"operation": "attach_asset", "product_present": False}, "deny", "asset_product_required", "select_product"),
	_rule("asset_url_required", "Assets require a URL.", {"operation": "attach_asset", "url_present": False}, "deny", "asset_url_required", "provide_url"),
	_rule("asset_rights_required", "Assets require rights basis.", {"operation": "attach_asset", "rights_basis_present": False}, "deny", "asset_rights_basis_required", "provide_rights_basis"),
	_rule("compliance_product_required", "Compliance records require a product.", {"operation": "record_compliance", "product_present": False}, "deny", "compliance_product_required", "select_product"),
	_rule("compliance_evidence_required", "Compliance records require evidence.", {"operation": "record_compliance", "evidence_present": False}, "deny", "compliance_evidence_required", "attach_evidence"),
	_rule("compliance_high_risk_review", "High-risk compliance requires review.", {"operation": "record_compliance", "high_risk": True, "review_recorded": False}, "require_review", "compliance_review_required", "record_compliance_review"),
	_rule("channel_product_required", "Channel listings require a product.", {"operation": "create_channel_listing", "product_present": False}, "deny", "channel_product_required", "select_product"),
	_rule("channel_supported", "Channel must be supported.", {"operation": "create_channel_listing", "channel_supported": False}, "deny", "channel_not_supported", "choose_supported_channel"),
	_rule("channel_listing_required", "Channel listings require listing id.", {"operation": "create_channel_listing", "listing_id_present": False}, "deny", "channel_listing_id_required", "provide_listing_id"),
	_rule("channel_approval_required", "Channel listings require approval.", {"operation": "create_channel_listing", "approval_recorded": False}, "deny", "channel_approval_required", "record_channel_approval"),
	_rule("publish_product_required", "Publishing requires a product.", {"operation": "publish_product", "product_present": False}, "deny", "publish_product_required", "select_product"),
	_rule("publish_content_required", "Publishing requires approved content.", {"operation": "publish_product", "approved_content_present": False}, "deny", "approved_content_required", "approve_content"),
	_rule("publish_channel_required", "Publishing requires approved channel.", {"operation": "publish_product", "approved_channel_present": False}, "deny", "approved_channel_required", "approve_channel"),
	_rule("publish_approval_required", "Publishing requires approval.", {"operation": "publish_product", "approval_recorded": False}, "deny", "publish_approval_required", "record_publish_approval"),
	_rule("quality_product_required", "Quality issues require a product.", {"operation": "record_quality_issue", "product_present": False}, "deny", "quality_product_required", "select_product"),
	_rule("quality_owner_required", "High or critical quality issues require an owner.", {"operation": "record_quality_issue", "high_or_critical": True, "owner_present": False}, "deny", "quality_owner_required", "assign_owner"),
	_rule("change_product_required", "Change requests require a product.", {"operation": "create_change_request", "product_present": False}, "deny", "change_product_required", "select_product"),
	_rule("change_reason_required", "Change requests require a reason.", {"operation": "create_change_request", "reason_present": False}, "deny", "change_reason_required", "provide_reason"),
	_rule("change_approval_required", "Change approval requires an approver.", {"operation": "approve_change", "approval_recorded": False}, "deny", "change_approval_required", "record_approver"),
	_rule("bytewax_event_stream_required", "PIM batches must use Bytewax event stream metadata.", {"operation": "pim_batch", "event_stream": "queue"}, "deny", "bytewax_event_stream_required", "route_to_bytewax_stream"),
	_rule("agent_runtime_supported", "PIM agents must use a supported runtime.", {"operation": "register_pim_agent", "runtime_supported": False}, "deny", "pim_agent_runtime_not_supported", "choose_supported_runtime"),
	_rule("agent_role_supported", "PIM agents must use a supported role.", {"operation": "register_pim_agent", "role_supported": False}, "deny", "pim_agent_role_not_supported", "choose_supported_role"),
	_rule("agent_scope_limited", "PIM agents cannot autonomously post privileged state changes.", {"operation": "agent_action", "privileged_action": True, "human_approved": False}, "require_review", "pim_agent_human_approval_required", "record_human_approval"),
	_rule("audit_required_for_state_change", "PIM state changes must be auditable.", {"operation_type": "write", "audit_enabled": False}, "deny", "pim_audit_required", "enable_audit"),
]


CONFIGURATION_SCHEMA = {"type": "object", "required": ["tenant_id", "ui", "theme"], "properties": {key: {"type": "object"} for key in DEFAULT_CONFIGURATION if key != "tenant_id"} | {"tenant_id": {"type": "string"}}}


def _merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
	merged = deepcopy(base)
	for key, value in overrides.items():
		if isinstance(value, dict) and isinstance(merged.get(key), dict):
			merged[key] = _merge_dict(merged[key], value)
		else:
			merged[key] = deepcopy(value)
	return merged


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	configuration = _merge_dict(DEFAULT_CONFIGURATION, overrides or {})
	configuration["tenant_id"] = tenant_id or configuration.get("tenant_id", "default")
	return {"capability": CAPABILITY_ID, "display_name": CAPABILITY_NAME, "name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": deepcopy(PROVIDES), "requires": deepcopy(REQUIRES), "configuration": configuration, "configuration_schema": deepcopy(CONFIGURATION_SCHEMA), "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/pde/pim/api/v1", "requires_theme": True, "template_roots": ["templates/", "static/"], "view_module": "views.py", "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if context.get(key) != expected:
			return False
	return True


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched_rules: list[str] = []
	effects: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched_rules.append(rule["name"])
			effect = deepcopy(rule["effect"])
			effects.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched_rules, "effects": effects, "context": deepcopy(context)}
