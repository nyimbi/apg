"""Executable capability contract for APG ITSM CMDB (Configuration Management Database)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CAPABILITY_ID = "itsm_cmdb"
CAPABILITY_NAME = "Configuration Management Database"
CAPABILITY_VERSION = "1.0.0"
CMDB_EVENT_STREAM = "apg.itsm.cmdb.lifecycle"

SUPPORTED_CI_TYPES = [
	"server", "vm", "container", "network_device", "storage", "database",
	"application", "service", "software_license", "certificate", "cloud_resource",
	"endpoint", "printer", "iot_device", "middleware", "api", "cluster",
]
SUPPORTED_CI_STATUSES = [
	"active", "inactive", "decommissioned", "maintenance", "planned",
	"retired", "unknown",
]
SUPPORTED_RELATIONSHIP_TYPES = [
	"depends_on", "hosts", "runs_on", "connects_to", "managed_by",
	"owned_by", "replicates", "backs_up", "monitors", "provides_service_to",
	"clustered_with", "virtualizes",
]
SUPPORTED_DISCOVERY_METHODS = [
	"network_scan", "agent_based", "api_poll", "snmp", "wmi",
	"ssh", "cloud_api", "manual", "import",
]
SUPPORTED_ENVIRONMENTS = ["production", "staging", "development", "testing", "dr"]
SUPPORTED_CHANGE_RECORD_STATUSES = ["pending", "approved", "applied", "failed", "rolled_back"]
SUPPORTED_HEALTH_STATUSES = ["healthy", "degraded", "critical", "unknown"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"ci": {
		"supported_ci_types": SUPPORTED_CI_TYPES,
		"supported_statuses": SUPPORTED_CI_STATUSES,
		"supported_environments": SUPPORTED_ENVIRONMENTS,
		"owner_required": True,
		"environment_required": True,
	},
	"relationships": {
		"supported_types": SUPPORTED_RELATIONSHIP_TYPES,
		"ci_required": True,
		"bidirectional_tracking": True,
	},
	"discovery": {
		"supported_methods": SUPPORTED_DISCOVERY_METHODS,
		"schedule_required": False,
		"auto_reconcile": True,
	},
	"change_tracking": {
		"supported_statuses": SUPPORTED_CHANGE_RECORD_STATUSES,
		"ci_required": True,
		"approver_required": True,
	},
	"health": {
		"supported_statuses": SUPPORTED_HEALTH_STATUSES,
		"score_range": [0, 100],
		"auto_compute": True,
	},
	"governance": {
		"require_tenant_context": True,
		"audit_events": True,
		"cross_tenant_denied": True,
		"decommission_requires_approval": True,
	},
	"observability": {
		"event_stream": CMDB_EVENT_STREAM,
		"stream_processor": "bytewax",
	},
	"adapters": {
		"auth": "auth",
		"audit": "audl",
		"discovery": "disc",
		"graph": "grph",
		"notifications": "ntfy",
	},
	"ui": {
		"enable_ci_registry": True,
		"enable_relationships": True,
		"enable_discovery": True,
		"enable_change_tracking": True,
		"enable_health_dashboard": True,
		"enable_dependency_map": True,
	},
	"theme": {
		"default_theme": "itsm_cmdb_control",
		"allow_tenant_overrides": True,
	},
}

PROVIDES = [
	"cmdb_ci_registry",
	"cmdb_relationship_graph",
	"cmdb_discovery_workflow",
	"cmdb_change_tracking",
	"cmdb_health_scoring",
	"cmdb_dependency_map",
]
REQUIRES = ["auth", "audl", "disc", "grph", "ntfy"]

UI_ROUTES = [
	{"name": "dashboard", "path": "/itsm-cmdb/dashboard", "component": "CmdbDashboard", "permission": "itsm_cmdb:view", "nav_group": "Overview"},
	{"name": "ci_registry", "path": "/itsm-cmdb/cis", "component": "CiRegistry", "permission": "itsm_cmdb:ci", "nav_group": "Assets"},
	{"name": "relationships", "path": "/itsm-cmdb/relationships", "component": "RelationshipGraph", "permission": "itsm_cmdb:relationships", "nav_group": "Assets"},
	{"name": "dependency_map", "path": "/itsm-cmdb/dependency-map", "component": "DependencyMap", "permission": "itsm_cmdb:view", "nav_group": "Assets"},
	{"name": "discovery", "path": "/itsm-cmdb/discovery", "component": "DiscoveryConsole", "permission": "itsm_cmdb:discovery", "nav_group": "Automation"},
	{"name": "change_tracking", "path": "/itsm-cmdb/changes", "component": "CmdbChangeLog", "permission": "itsm_cmdb:changes", "nav_group": "Governance"},
	{"name": "health", "path": "/itsm-cmdb/health", "component": "CmdbHealthDashboard", "permission": "itsm_cmdb:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/itsm-cmdb/settings", "component": "CmdbSettings", "permission": "itsm_cmdb:admin", "nav_group": "Administration"},
]

THEME = {
	"name": "itsm_cmdb_control",
	"tokens": {
		"color.primary": "#1D4ED8",
		"color.accent": "#7C3AED",
		"color.success": "#166534",
		"color.warning": "#A16207",
		"color.danger": "#991B1B",
		"surface.canvas": "#F8FAFC",
		"surface.panel": "#FFFFFF",
		"text.primary": "#111827",
		"text.secondary": "#4B5563",
		"border.radius": "8px",
		"density": "comfortable",
	},
	"components": {
		"ci_registry": {"icon": "server", "status_indicator": "ci-status-chip"},
		"relationships": {"icon": "git-branch", "status_indicator": "relation-chip"},
		"discovery": {"icon": "scan", "status_indicator": "discovery-chip"},
		"change_tracking": {"icon": "history", "status_indicator": "change-chip"},
		"health": {"icon": "activity", "status_indicator": "health-chip"},
	},
}

STREAMING = {
	"processor": "bytewax",
	"stream": CMDB_EVENT_STREAM,
	"key": "tenant_id",
	"events": [
		"ci_registered", "ci_updated", "ci_decommissioned",
		"relationship_created", "relationship_removed",
		"discovery_job_started", "discovery_job_completed",
		"change_record_created", "change_record_applied",
		"health_score_updated",
	],
	"guardrails": [
		"cross_tenant_ci_access_denied",
		"decommission_requires_approval",
		"undiscovered_ci_manual_only",
	],
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "ci_type_supported", "condition": {"operation": "register_ci", "ci_type_supported": False}, "effect": {"decision": "deny", "reason": "ci_type_not_supported", "required_action": "select_supported_ci_type"}},
	{"name": "ci_owner_required", "condition": {"operation": "register_ci", "owner_present": False}, "effect": {"decision": "deny", "reason": "ci_owner_required", "required_action": "assign_ci_owner"}},
	{"name": "ci_environment_required", "condition": {"operation": "register_ci", "environment_present": False}, "effect": {"decision": "deny", "reason": "ci_environment_required", "required_action": "set_ci_environment"}},
	{"name": "relationship_type_supported", "condition": {"operation": "create_relationship", "relationship_type_supported": False}, "effect": {"decision": "deny", "reason": "relationship_type_not_supported", "required_action": "select_supported_relationship_type"}},
	{"name": "relationship_ci_required", "condition": {"operation": "create_relationship", "source_ci_present": False}, "effect": {"decision": "deny", "reason": "source_ci_required", "required_action": "select_source_ci"}},
	{"name": "discovery_method_supported", "condition": {"operation": "create_discovery_job", "discovery_method_supported": False}, "effect": {"decision": "deny", "reason": "discovery_method_not_supported", "required_action": "select_supported_discovery_method"}},
	{"name": "decommission_requires_approval", "condition": {"operation": "decommission_ci", "approval_present": False}, "effect": {"decision": "deny", "reason": "decommission_approval_required", "required_action": "obtain_decommission_approval"}},
	{"name": "change_record_ci_required", "condition": {"operation": "record_ci_change", "ci_present": False}, "effect": {"decision": "deny", "reason": "ci_required_for_change_record", "required_action": "select_ci"}},
	{"name": "cross_tenant_ci_access_denied", "condition": {"operation": "access_ci", "cross_tenant": True}, "effect": {"decision": "deny", "reason": "cross_tenant_access_denied", "required_action": "use_tenant_scoped_context"}},
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
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"api_prefix": "/itsm-cmdb/api/v1",
			"requires_theme": True,
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
