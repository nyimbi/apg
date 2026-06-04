"""Executable capability contract for APG central configuration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_CONFIG_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_CONFIG_AGENT_ROLES = [
	"config_architect",
	"schema_reviewer",
	"release_reviewer",
	"drift_reviewer",
	"security_reviewer",
	"rollback_reviewer",
]
SUPPORTED_ENVIRONMENTS = ["development", "staging", "production", "dr", "sandbox"]
SUPPORTED_CONFIG_VALUE_TYPES = ["string", "integer", "float", "boolean", "json", "yaml", "secret_ref"]
SUPPORTED_DEPLOYMENT_STRATEGIES = ["immediate", "canary", "blue_green", "rolling", "scheduled"]
SUPPORTED_DRIFT_SEVERITIES = ["info", "warning", "critical"]
SUPPORTED_ROLLBACK_REASONS = ["deployment_failure", "drift_detected", "security_incident", "manual_revert"]
SUPPORTED_CIRCUIT_BREAKER_STATES = ["closed", "open", "half_open"]
SUPPORTED_TEMPLATE_SCOPES = ["private", "team", "shared", "global"]
SUPPORTED_AUDIT_LEVELS = ["none", "summary", "detailed", "forensic"]
SUPPORTED_IMPACT_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_VERSION_STRATEGIES = ["semver", "timestamp", "sequential"]
SUPPORTED_SECRET_BACKENDS = ["vault", "aws_secrets_manager", "azure_keyvault", "gcp_secret_manager"]
SUPPORTED_NOTIFICATION_CHANNELS = ["email", "slack", "pagerduty", "webhook", "sms"]
SUPPORTED_VALIDATION_MODES = ["strict", "lenient", "dry_run"]

CONFIG_EVENT_STREAM = "apg.composition.config.lifecycle"

PROVIDES = [
	"configuration_namespace_registry",
	"configuration_value_lifecycle",
	"configuration_schema_validation",
	"configuration_release_workflows",
	"configuration_template_library",
	"configuration_drift_monitoring",
	"config_agents",
	"cross_tenant_config_isolation",
	"circuit_breaker_config_gate",
	"cascading_config_failure_containment",
	"config_change_audit_trail",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_access",
	"composition_events",
	"moni",
	"conf",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"namespaces": {
		"owner_required": True,
		"environment_required": True,
		"capability_boundary_required": True,
		"path_prefix_required": True,
		"cross_tenant_isolation_enforced": True,
	},
	"configurations": {
		"schema_required_for_restricted": True,
		"secret_reference_required": True,
		"validation_required_before_activation": True,
		"versioning_enabled": True,
		"drift_detection_enabled": True,
		"value_type_required": True,
	},
	"deployments": {
		"approval_required_for_production": True,
		"canary_required_for_high_impact": True,
		"rollback_supported": True,
		"deployment_stream_required": True,
		"max_simultaneous_deployments_per_tenant": 3,
		"blast_radius_estimation_required": True,
	},
	"templates": {
		"owner_required": True,
		"variable_schema_required": True,
		"review_required_for_shared": True,
		"scope_required": True,
	},
	"circuit_breaker": {
		"enabled": True,
		"failure_threshold": 5,
		"recovery_timeout_seconds": 60,
		"half_open_probe_count": 2,
		"cascade_isolation_enabled": True,
		"config_deployment_gate_enabled": True,
	},
	"cascading_failure": {
		"dependency_health_check_enabled": True,
		"fallback_config_required": True,
		"bulkhead_isolation_enabled": True,
		"max_downstream_config_failures": 3,
		"quarantine_namespace_on_cascade": True,
	},
	"config_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_CONFIG_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_CONFIG_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_and_validate",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"redact_secret_values": True,
		"policy_attached_for_writes": True,
		"privilege_escalation_blocked": True,
		"cross_tenant_write_blocked": True,
	},
	"observability": {
		"event_stream": CONFIG_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_namespace_events": True,
		"emit_config_events": True,
		"emit_deployment_events": True,
		"emit_drift_events": True,
		"emit_circuit_breaker_events": True,
		"emit_cascade_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"secrets": "adapter",
		"notification": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"monitoring": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_namespace_console": True,
		"enable_config_editor": True,
		"enable_release_board": True,
		"enable_template_library": True,
		"enable_drift_monitor": True,
		"enable_circuit_breaker_console": True,
		"enable_blast_radius_analyzer": True,
		"enable_agent_workbench": True,
		"enable_audit_console": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_config_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"namespaces",
		"configurations",
		"deployments",
		"templates",
		"circuit_breaker",
		"cascading_failure",
		"config_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"namespaces": {"type": "object"},
		"configurations": {"type": "object"},
		"deployments": {"type": "object"},
		"templates": {"type": "object"},
		"circuit_breaker": {"type": "object"},
		"cascading_failure": {"type": "object"},
		"config_agents": {"type": "object"},
		"governance": {"type": "object"},
		"observability": {"type": "object"},
		"adapters": {"type": "object"},
		"ui": {"type": "object"},
		"theme": {"type": "object"},
	},
}

RULES: list[dict[str, Any]] = [
	# --- Tenant context (hard gate) ---
	{
		"name": "tenant_context_required",
		"description": "All central-configuration operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	# --- Write-requires-policy ---
	{
		"name": "configuration_requires_policy",
		"description": "Configuration writes require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	# --- Cross-tenant isolation ---
	{
		"name": "cross_tenant_config_write_blocked",
		"description": "Configuration writes may not target a namespace owned by a different tenant.",
		"condition": {"cross_tenant_write_attempted": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_config_write_forbidden", "required_action": "reject_cross_tenant_config_write"},
	},
	{
		"name": "cross_tenant_template_reference_blocked",
		"description": "Templates may not reference variables or namespaces from a different tenant.",
		"condition": {"operation": "create_template", "cross_tenant_reference_present": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_template_reference_forbidden", "required_action": "remove_cross_tenant_template_reference"},
	},
	# --- Privilege escalation prevention ---
	{
		"name": "config_privilege_escalation_blocked",
		"description": "A principal may not deploy to an environment beyond their authorised scope.",
		"condition": {"operation": "deploy_configuration", "privilege_escalation_detected": True},
		"effect": {"decision": "deny", "reason": "config_privilege_escalation_forbidden", "required_action": "request_elevated_deploy_approval"},
	},
	# --- Circuit breaker rules ---
	{
		"name": "circuit_breaker_open_blocks_deployments",
		"description": "When the config circuit breaker is open, no new deployments are permitted.",
		"condition": {"circuit_breaker_state": "open", "operation": "deploy_configuration"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_open", "required_action": "wait_for_circuit_recovery"},
	},
	{
		"name": "circuit_breaker_half_open_limits_deployments",
		"description": "In half-open state only a single probe deployment is permitted per tenant.",
		"condition": {"circuit_breaker_state": "half_open", "probe_budget_exhausted": True},
		"effect": {"decision": "deny", "reason": "circuit_breaker_half_open_budget_exhausted", "required_action": "shed_deployment_load"},
	},
	{
		"name": "circuit_breaker_trip_requires_event",
		"description": "Config circuit breaker state transitions must emit a Bytewax event.",
		"condition": {"operation": "trip_circuit_breaker", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_event_required", "required_action": "emit_circuit_breaker_event_to_bytewax"},
	},
	# --- Cascading failure containment ---
	{
		"name": "cascade_isolation_on_downstream_config_failure",
		"description": "When downstream config failures exceed threshold, quarantine the namespace.",
		"condition": {"downstream_failure_count_gt": 3, "namespace_quarantine_active": False},
		"effect": {"decision": "require_review", "reason": "namespace_cascade_isolation_required", "required_action": "quarantine_namespace"},
	},
	{
		"name": "bulkhead_overflow_blocks_config_deploy",
		"description": "Deployments exceeding the per-tenant simultaneous limit are denied.",
		"condition": {"operation": "deploy_configuration", "simultaneous_deployment_count_gt": 3},
		"effect": {"decision": "deny", "reason": "simultaneous_deployment_limit_exceeded", "required_action": "queue_deployment"},
	},
	{
		"name": "fallback_config_required_for_degraded_dependency",
		"description": "When a config dependency is degraded a fallback config value must be active.",
		"condition": {"dependency_state": "degraded", "fallback_config_active": False},
		"effect": {"decision": "require_review", "reason": "fallback_config_required", "required_action": "activate_fallback_config"},
	},
	# --- Namespace lifecycle ---
	{
		"name": "namespace_requires_owner",
		"description": "Configuration namespaces require an accountable owner.",
		"condition": {"operation": "register_namespace", "namespace_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "namespace_owner_required", "required_action": "assign_namespace_owner"},
	},
	{
		"name": "namespace_requires_environment",
		"description": "Configuration namespaces require an environment.",
		"condition": {"operation": "register_namespace", "environment_present": False},
		"effect": {"decision": "deny", "reason": "namespace_environment_required", "required_action": "select_environment"},
	},
	# --- Configuration value lifecycle ---
	{
		"name": "restricted_configuration_requires_schema",
		"description": "Restricted configuration values require a schema.",
		"condition": {"operation": "create_configuration", "restricted_config": True, "schema_present": False},
		"effect": {"decision": "deny", "reason": "configuration_schema_required", "required_action": "attach_configuration_schema"},
	},
	{
		"name": "secret_configuration_requires_reference",
		"description": "Secret configuration values must use a secret reference.",
		"condition": {"operation": "create_configuration", "secret_config": True, "secret_reference_present": False},
		"effect": {"decision": "deny", "reason": "secret_reference_required", "required_action": "attach_secret_reference"},
	},
	{
		"name": "activation_requires_validation",
		"description": "Configurations require validation evidence before activation.",
		"condition": {"operation": "activate_configuration", "validation_evidence_present": False},
		"effect": {"decision": "deny", "reason": "configuration_validation_required", "required_action": "attach_validation_evidence"},
	},
	# --- Deployment lifecycle ---
	{
		"name": "production_deployment_requires_approval",
		"description": "Production deployments require approval.",
		"condition": {"operation": "deploy_configuration", "environment": "production", "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "production_deployment_approval_required", "required_action": "record_deployment_approval"},
	},
	{
		"name": "high_impact_deployment_requires_canary",
		"description": "High-impact configuration deployments require canary evidence.",
		"condition": {"operation": "deploy_configuration", "impact_level": "high", "canary_evidence_present": False},
		"effect": {"decision": "require_review", "reason": "canary_evidence_required", "required_action": "attach_canary_evidence"},
	},
	{
		"name": "high_impact_deployment_requires_blast_radius",
		"description": "High-impact deployments must have a blast-radius estimate attached.",
		"condition": {"operation": "deploy_configuration", "impact_level": "high", "blast_radius_estimated": False},
		"effect": {"decision": "deny", "reason": "blast_radius_estimation_required", "required_action": "attach_blast_radius_estimate"},
	},
	{
		"name": "deployment_requires_bytewax_stream",
		"description": "Configuration deployment events must be emitted through Bytewax.",
		"condition": {"operation": "deploy_configuration", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_config_deployment_to_bytewax"},
	},
	# --- Rollback lifecycle ---
	{
		"name": "rollback_requires_reason",
		"description": "Configuration rollback requires a reason.",
		"condition": {"operation": "rollback_configuration", "rollback_reason_present": False},
		"effect": {"decision": "deny", "reason": "rollback_reason_required", "required_action": "attach_rollback_reason"},
	},
	{
		"name": "rollback_requires_bytewax_stream",
		"description": "Configuration rollback events must be emitted through Bytewax.",
		"condition": {"operation": "rollback_configuration", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_config_rollback_to_bytewax"},
	},
	# --- Template lifecycle ---
	{
		"name": "shared_template_requires_review",
		"description": "Shared templates require review.",
		"condition": {"operation": "create_template", "shared_template": True, "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "shared_template_review_required", "required_action": "record_template_review"},
	},
	# --- Batch / streaming ---
	{
		"name": "batch_change_requires_bytewax",
		"description": "Batch configuration changes require Bytewax coordination.",
		"condition": {"operation": "batch_configuration_change", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_config_change_to_bytewax"},
	},
	# --- Agent governance ---
	{
		"name": "config_agent_runtime_supported",
		"description": "Configuration agents must use an approved runtime.",
		"condition": {"operation": "register_config_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "config_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "config_agent_role_supported",
		"description": "Configuration agents must use an approved role.",
		"condition": {"operation": "register_config_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "config_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_agent_config_action_requires_human_approval",
		"description": "Privileged configuration actions proposed by agents require human approval.",
		"condition": {"operation": "agent_config_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# --- Service mesh integrity ---
	{
		"name": "service_mesh_identity_required_for_config_read",
		"description": "Intra-mesh callers reading restricted config must present a verified mesh identity.",
		"condition": {"operation": "read_restricted_config", "mesh_identity_verified": False},
		"effect": {"decision": "deny", "reason": "mesh_identity_required", "required_action": "attach_verified_mesh_identity"},
	},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-config/dashboard", "component": "ConfigDashboard", "permission": "composition_config:view", "nav_group": "Overview"},
	{"name": "namespaces", "path": "/composition-config/namespaces", "component": "ConfigNamespaceConsole", "permission": "composition_config:admin", "nav_group": "Namespaces"},
	{"name": "configurations", "path": "/composition-config/configurations", "component": "ConfigEditor", "permission": "composition_config:edit", "nav_group": "Configuration"},
	{"name": "releases", "path": "/composition-config/releases", "component": "ConfigReleaseBoard", "permission": "composition_config:release", "nav_group": "Release"},
	{"name": "templates", "path": "/composition-config/templates", "component": "ConfigTemplateLibrary", "permission": "composition_config:edit", "nav_group": "Configuration"},
	{"name": "drift", "path": "/composition-config/drift", "component": "ConfigDriftMonitor", "permission": "composition_config:operate", "nav_group": "Operations"},
	{"name": "blast_radius", "path": "/composition-config/blast-radius", "component": "ConfigBlastRadiusAnalyzer", "permission": "composition_config:govern", "nav_group": "Risk"},
	{"name": "circuit_breaker", "path": "/composition-config/circuit-breaker", "component": "ConfigCircuitBreakerConsole", "permission": "composition_config:operate", "nav_group": "Resilience"},
	{"name": "agents", "path": "/composition-config/agents", "component": "ConfigAgentWorkbench", "permission": "composition_config:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/composition-config/audit", "component": "ConfigAuditConsole", "permission": "composition_config:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/composition-config/settings", "component": "ConfigSettings", "permission": "composition_config:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_config_control",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#C44536",
		"color.success": "#2F855A",
		"color.warning": "#B7791F",
		"color.danger": "#C53030",
		"surface.canvas": "#F7F8FA",
		"surface.panel": "#FFFFFF",
		"text.primary": "#172033",
		"text.secondary": "#52606D",
		"border.radius": "8px",
		"density": "compact",
	},
	"components": {
		"namespace_console": {"icon": "folder-tree", "status_indicator": "namespace-pill", "risk_style": "boundary-band"},
		"config_editor": {"visual": "schema-editor", "status_style": "version-chip"},
		"release_board": {"visual": "release-lanes", "status_style": "approval-chip"},
		"template_library": {"visual": "template-grid", "status_style": "template-chip"},
		"drift_monitor": {"visual": "diff-timeline", "status_style": "drift-chip"},
		"blast_radius_analyzer": {"visual": "impact-heatmap", "status_style": "impact-chip"},
		"circuit_breaker_console": {"visual": "breaker-gauge", "status_style": "breaker-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_config",
		"display_name": "Central Configuration Management",
		"version": "1.2.0",
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/composition-config/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"streaming": streaming_manifest(),
	}


def streaming_manifest() -> dict[str, Any]:
	return {
		"processor": "bytewax",
		"stream": CONFIG_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"namespace_registered",
			"namespace_quarantined",
			"configuration_created",
			"configuration_validated",
			"configuration_activated",
			"configuration_deployed",
			"configuration_rolled_back",
			"deployment_canary_started",
			"deployment_blast_radius_estimated",
			"drift_detected",
			"drift_remediated",
			"circuit_breaker_tripped",
			"circuit_breaker_recovered",
			"cascade_isolation_triggered",
			"template_created",
			"template_review_completed",
			"config_agent_registered",
		],
		"states": ["draft", "validated", "active", "release_pending", "deployed", "rolled_back", "drifted", "quarantined", "blocked"],
		"guardrails": [
			"deployment_requires_bytewax_stream",
			"rollback_requires_bytewax_stream",
			"batch_change_requires_bytewax",
			"privileged_agent_config_action_requires_human_approval",
			"circuit_breaker_trip_requires_event",
			"cross_tenant_config_write_blocked",
		],
	}


def event_stream_name() -> str:
	return CONFIG_EVENT_STREAM


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	matched: list[str] = []
	actions: list[dict[str, Any]] = []
	decision = "allow"
	for rule in RULES:
		if _matches(rule["condition"], context):
			matched.append(rule["name"])
			effect = rule["effect"]
			actions.append(effect)
			if effect["decision"] == "deny":
				decision = "deny"
			elif effect["decision"] == "require_review" and decision != "deny":
				decision = "require_review"
	return {"decision": decision, "matched_rules": matched, "actions": actions, "context": context}


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
	for key, expected in condition.items():
		if key.endswith("_lte"):
			if not context.get(key[:-4], 0) <= expected:
				return False
		elif key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gte"):
			if not context.get(key[:-4], 0) >= expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
				return False
		elif key.endswith("_ne"):
			if context.get(key[:-3]) == expected:
				return False
		elif context.get(key) != expected:
			return False
	return True


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
	for key, value in source.items():
		if isinstance(value, dict) and isinstance(target.get(key), dict):
			_deep_merge(target[key], value)
		else:
			target[key] = value
