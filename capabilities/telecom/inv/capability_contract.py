"""Executable capability contract for APG Network Inventory."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "telecom_inv"
CAPABILITY_NAME = "Network Inventory"
CAPABILITY_VERSION = "1.0.0"
INV_EVENT_STREAM = "apg.telecom.inv.lifecycle"

SUPPORTED_ASSET_TYPES = ["base_station", "antenna", "transmission_link", "router", "switch", "server", "cable", "fibre_duct", "power_unit", "cooling_unit", "shelter", "site"]
SUPPORTED_CIRCUIT_TYPES = ["e1_t1", "stm1", "stm4", "stm16", "ethernet_10g", "ethernet_100g", "wavelength", "mpls_lsp", "pseudowire", "vpn_l2", "vpn_l3"]
SUPPORTED_IP_VERSIONS = ["ipv4", "ipv6", "dual_stack"]
SUPPORTED_ADDRESS_BLOCK_TYPES = ["loopback", "point_to_point", "lan_subnet", "management", "customer_pool", "transit", "anycast"]
SUPPORTED_TOPOLOGY_TYPES = ["ring", "mesh", "star", "hub_spoke", "hierarchical", "flat", "point_to_point"]
SUPPORTED_ASSET_STATUSES = ["planned", "ordered", "in_transit", "installed", "commissioned", "active", "degraded", "decommissioned", "disposed"]
SUPPORTED_CIRCUIT_STATUSES = ["planned", "provisioned", "active", "suspended", "failed", "decommissioned"]
SUPPORTED_NETWORK_DOMAINS = ["core", "metro", "access", "backhaul", "enterprise_edge", "data_centre", "ocs", "ims"]
SUPPORTED_VENDOR_TYPES = ["ericsson", "nokia", "huawei", "cisco", "juniper", "samsung", "zte", "other"]
SUPPORTED_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_AGENT_ROLES = ["inventory_auditor", "circuit_planner", "ip_manager", "topology_mapper", "reconciliation_agent"]
SUPPORTED_REVIEW_STATUSES = ["approved", "rejected", "needs_changes", "escalated"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"assets": {"supported_types": SUPPORTED_ASSET_TYPES, "supported_statuses": SUPPORTED_ASSET_STATUSES, "supported_vendors": SUPPORTED_VENDOR_TYPES, "serial_number_required": True, "location_required": True},
	"circuits": {"supported_types": SUPPORTED_CIRCUIT_TYPES, "supported_statuses": SUPPORTED_CIRCUIT_STATUSES, "endpoint_required": True, "capacity_required": True},
	"ipam": {"supported_ip_versions": SUPPORTED_IP_VERSIONS, "supported_block_types": SUPPORTED_ADDRESS_BLOCK_TYPES, "prefix_length_required": True, "vrf_required": True},
	"topology": {"supported_types": SUPPORTED_TOPOLOGY_TYPES, "supported_domains": SUPPORTED_NETWORK_DOMAINS, "diagram_generation": True, "layer_visibility": True},
	"reconciliation": {"auto_discovery_enabled": True, "discrepancy_alerting": True, "approval_required": True},
	"agents": {"enabled": True, "supported_runtimes": SUPPORTED_AGENT_RUNTIMES, "supported_roles": SUPPORTED_AGENT_ROLES, "human_approval_required_for_privileged_actions": True},
	"governance": {"require_tenant_context": True, "policy_attached_for_writes": True, "audit_events": True, "unauthorised_decommission_denied": True, "cross_tenant_inventory_denied": True},
	"observability": {"event_stream": INV_EVENT_STREAM, "stream_processor": "bytewax"},
	"ui": {"enable_dashboard": True, "enable_assets": True, "enable_circuits": True, "enable_ipam": True, "enable_topology": True, "enable_reconciliation": True, "enable_agents": True},
	"theme": {"default_theme": "telecom_inv_control", "allow_tenant_overrides": True},
}

PROVIDES = ["asset_inventory_workflow", "circuit_management_workflow", "ipam_workflow", "topology_documentation_workflow", "inventory_reconciliation_workflow", "network_resource_query", "inv_agent_workflow"]
REQUIRES = ["auth", "audl", "mten", "conf", "nlpc", "moni", "mqeb", "geos"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/telecom-inv/dashboard", "component": "InvDashboard", "permission": "telecom_inv:view", "nav_group": "Overview"},
	{"name": "assets", "path": "/telecom-inv/assets", "component": "InvAssetConsole", "permission": "telecom_inv:assets", "nav_group": "Physical"},
	{"name": "asset_detail", "path": "/telecom-inv/assets/<id>", "component": "InvAssetDetail", "permission": "telecom_inv:assets", "nav_group": "Physical"},
	{"name": "circuits", "path": "/telecom-inv/circuits", "component": "InvCircuitConsole", "permission": "telecom_inv:circuits", "nav_group": "Logical"},
	{"name": "ipam", "path": "/telecom-inv/ipam", "component": "InvIpamConsole", "permission": "telecom_inv:ipam", "nav_group": "Logical"},
	{"name": "topology", "path": "/telecom-inv/topology", "component": "InvTopologyViewer", "permission": "telecom_inv:topology", "nav_group": "Topology"},
	{"name": "sites", "path": "/telecom-inv/sites", "component": "InvSiteConsole", "permission": "telecom_inv:assets", "nav_group": "Physical"},
	{"name": "reconciliation", "path": "/telecom-inv/reconciliation", "component": "InvReconciliationConsole", "permission": "telecom_inv:reconciliation", "nav_group": "Operations"},
	{"name": "discrepancies", "path": "/telecom-inv/discrepancies", "component": "InvDiscrepancyQueue", "permission": "telecom_inv:reconciliation", "nav_group": "Operations"},
	{"name": "agents", "path": "/telecom-inv/agents", "component": "InvAgentWorkbench", "permission": "telecom_inv:admin", "nav_group": "Automation"},
	{"name": "settings", "path": "/telecom-inv/settings", "component": "InvSettings", "permission": "telecom_inv:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "telecom_inv_control",
	"tokens": {"color.primary": "#92400E", "color.accent": "#0369A1", "color.success": "#15803D", "color.warning": "#B45309", "color.danger": "#B91C1C", "surface.canvas": "#F8FAFC", "surface.panel": "#FFFFFF", "text.primary": "#111827", "text.secondary": "#4B5563", "border.radius": "8px", "density": "compact"},
	"components": {"assets": {"icon": "server", "status_indicator": "asset-status-chip"}, "circuits": {"icon": "git-branch", "status_indicator": "circuit-status-chip"}, "ipam": {"icon": "globe", "status_indicator": "ip-version-chip"}, "topology": {"icon": "share-2", "status_indicator": "topology-type-chip"}, "sites": {"icon": "map-pin", "status_indicator": "site-chip"}, "reconciliation": {"icon": "refresh-cw", "status_indicator": "reconcile-chip"}, "discrepancies": {"icon": "alert-triangle", "status_indicator": "discrepancy-chip"}, "agents": {"icon": "bot", "status_indicator": "agent-runtime-chip"}},
}

STREAMING = {"processor": "bytewax", "stream": INV_EVENT_STREAM, "key": "tenant_id", "events": ["asset_commissioned", "asset_decommissioned", "circuit_provisioned", "circuit_decommissioned", "ip_block_allocated", "ip_block_released", "topology_updated", "discrepancy_detected", "reconciliation_approved", "inv_agent_registered"], "guardrails": ["inv_batch_requires_bytewax", "privileged_inv_agent_action_requires_human_approval", "unauthorised_decommission_denied", "cross_tenant_inventory_denied"]}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "inv_write_requires_policy", "condition": {"operation_type": "write", "policy_attached": False}, "effect": {"decision": "deny", "reason": "inv_policy_required", "required_action": "attach_inv_policy"}},
	{"name": "asset_type_supported", "condition": {"operation": "commission_asset", "asset_type_supported": False}, "effect": {"decision": "deny", "reason": "asset_type_not_supported", "required_action": "select_supported_asset_type"}},
	{"name": "asset_serial_number_required", "condition": {"operation": "commission_asset", "serial_number_present": False}, "effect": {"decision": "deny", "reason": "serial_number_required", "required_action": "set_serial_number"}},
	{"name": "asset_location_required", "condition": {"operation": "commission_asset", "location_present": False}, "effect": {"decision": "deny", "reason": "asset_location_required", "required_action": "set_asset_location"}},
	{"name": "asset_status_supported", "condition": {"operation": "update_asset_status", "asset_status_supported": False}, "effect": {"decision": "deny", "reason": "asset_status_not_supported", "required_action": "select_supported_asset_status"}},
	{"name": "decommission_requires_approval", "condition": {"operation": "decommission_asset", "approval_present": False}, "effect": {"decision": "deny", "reason": "decommission_approval_required", "required_action": "attach_decommission_approval"}},
	{"name": "circuit_type_supported", "condition": {"operation": "provision_circuit", "circuit_type_supported": False}, "effect": {"decision": "deny", "reason": "circuit_type_not_supported", "required_action": "select_supported_circuit_type"}},
	{"name": "circuit_endpoint_required", "condition": {"operation": "provision_circuit", "endpoint_present": False}, "effect": {"decision": "deny", "reason": "circuit_endpoint_required", "required_action": "set_circuit_endpoints"}},
	{"name": "circuit_capacity_required", "condition": {"operation": "provision_circuit", "capacity_present": False}, "effect": {"decision": "deny", "reason": "circuit_capacity_required", "required_action": "set_circuit_capacity"}},
	{"name": "ip_version_supported", "condition": {"operation": "allocate_ip_block", "ip_version_supported": False}, "effect": {"decision": "deny", "reason": "ip_version_not_supported", "required_action": "select_supported_ip_version"}},
	{"name": "ip_block_prefix_required", "condition": {"operation": "allocate_ip_block", "prefix_length_present": False}, "effect": {"decision": "deny", "reason": "prefix_length_required", "required_action": "set_prefix_length"}},
	{"name": "ip_block_vrf_required", "condition": {"operation": "allocate_ip_block", "vrf_present": False}, "effect": {"decision": "deny", "reason": "vrf_required", "required_action": "set_vrf"}},
	{"name": "topology_type_supported", "condition": {"operation": "record_topology", "topology_type_supported": False}, "effect": {"decision": "deny", "reason": "topology_type_not_supported", "required_action": "select_supported_topology_type"}},
	{"name": "reconciliation_approval_required", "condition": {"operation": "approve_reconciliation", "approval_present": False}, "effect": {"decision": "deny", "reason": "reconciliation_approval_required", "required_action": "attach_reconciliation_approval"}},
	{"name": "unauthorised_decommission_denied", "condition": {"operation": "inv_agent_action", "unauthorised_decommission_scope": True}, "effect": {"decision": "deny", "reason": "unauthorised_decommission_denied", "required_action": "remove_decommission_scope"}},
	{"name": "cross_tenant_inventory_denied", "condition": {"operation": "inv_agent_action", "cross_tenant_inventory_scope": True}, "effect": {"decision": "deny", "reason": "cross_tenant_inventory_denied", "required_action": "remove_cross_tenant_scope"}},
	{"name": "inv_batch_requires_bytewax", "condition": {"operation": "inv_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_inv_batch_to_bytewax"}},
	{"name": "inv_agent_runtime_supported", "condition": {"operation": "register_inv_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "inv_agent_runtime_not_supported", "required_action": "select_supported_runtime"}},
	{"name": "inv_agent_role_supported", "condition": {"operation": "register_inv_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "inv_agent_role_not_supported", "required_action": "select_supported_role"}},
	{"name": "inv_agent_name_required", "condition": {"operation": "register_inv_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "inv_agent_name_required", "required_action": "name_inv_agent"}},
	{"name": "inv_agent_scope_required", "condition": {"operation": "register_inv_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "inv_agent_scope_required", "required_action": "bound_inv_agent_scope"}},
	{"name": "privileged_inv_agent_action_requires_human_approval", "condition": {"operation": "inv_agent_action", "privileged_scope": True, "human_approval_recorded": False}, "effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"}},
]


def get_capability_contract(tenant_id: str = "default") -> dict[str, Any]:
	configuration = deepcopy(DEFAULT_CONFIGURATION)
	configuration["tenant_id"] = tenant_id
	return {"capability": CAPABILITY_ID, "name": CAPABILITY_NAME, "display_name": CAPABILITY_NAME, "version": CAPABILITY_VERSION, "provides": list(PROVIDES), "requires": list(REQUIRES), "configuration": configuration, "configuration_schema": {"type": "object", "required": list(configuration), "properties": {key: {"type": "object"} for key in configuration if key != "tenant_id"} | {"tenant_id": {"type": "string", "minLength": 1}}}, "rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "api_prefix": "/telecom-inv/api/v1", "requires_theme": True, "view_module": "views.py", "template_roots": ["templates/", "static/"], "routes": deepcopy(UI_ROUTES)}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
