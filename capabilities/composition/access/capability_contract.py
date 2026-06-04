"""Executable capability contract for APG composition access control."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ACCESS_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ACCESS_AGENT_ROLES = [
	"access_architect",
	"policy_reviewer",
	"grant_reviewer",
	"risk_reviewer",
	"session_reviewer",
	"audit_reviewer",
]
SUPPORTED_IDENTITY_PROVIDER_TYPES = ["local", "oidc", "saml", "ldap", "api_key", "jwt"]
SUPPORTED_GRANT_SCOPES = ["read", "write", "admin", "privileged", "elevated", "read_and_recommend"]
SUPPORTED_POLICY_EFFECTS = ["allow", "deny", "require_mfa", "require_step_up", "require_approval"]
SUPPORTED_RESOURCE_SCOPES = ["public", "internal", "restricted", "confidential", "top_secret"]
SUPPORTED_SESSION_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_ACCESS_DECISION_OUTCOMES = ["allow", "deny", "step_up_required", "review_required", "blocked"]
SUPPORTED_CIRCUIT_BREAKER_STATES = ["closed", "open", "half_open"]
SUPPORTED_ENVIRONMENTS = ["development", "staging", "production", "dr"]
SUPPORTED_AUDIT_LEVELS = ["none", "summary", "detailed", "forensic"]
SUPPORTED_MFA_FACTORS = ["totp", "webauthn", "sms", "email", "hardware_key"]
SUPPORTED_TOKEN_TYPES = ["access", "refresh", "id", "service_account", "delegation"]
SUPPORTED_ACCESS_REVIEW_FREQUENCIES = ["monthly", "quarterly", "semi_annual", "annual"]

ACCESS_EVENT_STREAM = "apg.composition.access.lifecycle"

PROVIDES = [
	"identity_provider_composition",
	"resource_access_registry",
	"policy_orchestration",
	"grant_lifecycle",
	"session_risk_control",
	"access_decision_audit",
	"access_agents",
	"cross_tenant_isolation",
	"privilege_escalation_prevention",
	"circuit_breaker_enforcement",
	"cascading_failure_containment",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"conf",
	"composition_events",
	"moni",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"identity_providers": {
		"supported_types": SUPPORTED_IDENTITY_PROVIDER_TYPES,
		"provider_owner_required": True,
		"metadata_validation_required": True,
		"secret_reference_required": True,
		"test_evidence_required": True,
	},
	"resources": {
		"registry_required": True,
		"owner_required": True,
		"scope_required": True,
		"composition_boundary_required": True,
		"cross_tenant_isolation_enforced": True,
	},
	"policies": {
		"policy_owner_required": True,
		"effect_required": True,
		"conditions_required_for_sensitive_resources": True,
		"simulation_required_for_high_risk": True,
		"review_required_for_deny_override": True,
		"cross_tenant_policy_blocked": True,
	},
	"grants": {
		"approval_required_for_privileged": True,
		"separation_of_duties_required": True,
		"expiry_required_for_elevated_access": True,
		"justification_required": True,
		"max_grant_duration_days": 90,
		"periodic_review_required": True,
		"review_frequency": "quarterly",
	},
	"sessions": {
		"risk_scoring_enabled": True,
		"adaptive_step_up_enabled": True,
		"max_risk_without_review": 74,
		"continuous_evaluation_enabled": True,
		"max_session_duration_minutes": 480,
		"concurrent_session_limit": 5,
	},
	"circuit_breaker": {
		"enabled": True,
		"failure_threshold": 5,
		"recovery_timeout_seconds": 30,
		"half_open_probe_count": 3,
		"cascade_isolation_enabled": True,
		"tenant_isolation_enforced": True,
	},
	"cascading_failure": {
		"dependency_health_check_enabled": True,
		"fallback_policy_required": True,
		"bulkhead_isolation_enabled": True,
		"max_downstream_failures": 3,
		"quarantine_on_cascade_detected": True,
	},
	"access_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ACCESS_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ACCESS_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "read_and_recommend",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"cross_capability_guardrails": True,
		"privileged_action_review": True,
		"privilege_escalation_blocked": True,
		"periodic_access_review_enabled": True,
	},
	"observability": {
		"event_stream": ACCESS_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_provider_events": True,
		"emit_policy_events": True,
		"emit_grant_events": True,
		"emit_decision_events": True,
		"emit_circuit_breaker_events": True,
		"emit_cascade_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"identity": "adapter",
		"audit": "adapter",
		"notification": "adapter",
		"event_stream": "bytewax",
		"theme": "adapter",
		"monitoring": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_provider_console": True,
		"enable_policy_studio": True,
		"enable_grant_workbench": True,
		"enable_decision_explorer": True,
		"enable_agent_workbench": True,
		"enable_audit_console": True,
		"enable_circuit_breaker_console": True,
		"enable_access_review_console": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_access_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"identity_providers",
		"resources",
		"policies",
		"grants",
		"sessions",
		"circuit_breaker",
		"cascading_failure",
		"access_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"identity_providers": {"type": "object"},
		"resources": {"type": "object"},
		"policies": {"type": "object"},
		"grants": {"type": "object"},
		"sessions": {"type": "object"},
		"circuit_breaker": {"type": "object"},
		"cascading_failure": {"type": "object"},
		"access_agents": {"type": "object"},
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
		"description": "All composition access operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	# --- Write-requires-policy ---
	{
		"name": "access_write_requires_policy",
		"description": "Access write operations require an attached policy.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	# --- Cross-tenant isolation ---
	{
		"name": "cross_tenant_access_blocked",
		"description": "Principals may not access resources belonging to a different tenant.",
		"condition": {"cross_tenant_access_attempted": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_access_forbidden", "required_action": "reject_cross_tenant_request"},
	},
	{
		"name": "cross_tenant_policy_blocked",
		"description": "Policies must not reference resources or principals from a different tenant.",
		"condition": {"operation": "create_policy", "cross_tenant_reference_present": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_policy_forbidden", "required_action": "remove_cross_tenant_reference"},
	},
	# --- Privilege escalation prevention ---
	{
		"name": "privilege_escalation_blocked",
		"description": "A principal may not grant themselves a scope beyond their current maximum.",
		"condition": {"operation": "create_grant", "privilege_escalation_detected": True},
		"effect": {"decision": "deny", "reason": "privilege_escalation_forbidden", "required_action": "request_elevated_via_approval_flow"},
	},
	{
		"name": "grant_scope_exceeds_grantor_scope",
		"description": "A grantor cannot issue a grant with a scope exceeding their own authorised scope.",
		"condition": {"operation": "create_grant", "grant_scope_exceeds_grantor": True},
		"effect": {"decision": "deny", "reason": "grant_scope_exceeds_grantor_scope", "required_action": "reduce_grant_scope"},
	},
	# --- Circuit breaker rules ---
	{
		"name": "circuit_breaker_open_blocks_requests",
		"description": "When the access circuit breaker is open, all non-health-check requests are denied until recovery.",
		"condition": {"circuit_breaker_state": "open", "request_type_ne": "health_check"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_open", "required_action": "wait_for_circuit_recovery"},
	},
	{
		"name": "circuit_breaker_half_open_limits_throughput",
		"description": "In half-open state only probe requests are allowed; excess traffic is shed.",
		"condition": {"circuit_breaker_state": "half_open", "probe_budget_exhausted": True},
		"effect": {"decision": "deny", "reason": "circuit_breaker_half_open_budget_exhausted", "required_action": "shed_load_until_probe_completes"},
	},
	{
		"name": "circuit_breaker_trip_requires_event",
		"description": "Circuit breaker state transitions must emit a Bytewax lifecycle event.",
		"condition": {"operation": "trip_circuit_breaker", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_event_required", "required_action": "emit_circuit_breaker_event_to_bytewax"},
	},
	# --- Cascading failure containment ---
	{
		"name": "cascade_isolation_on_downstream_failure",
		"description": "When downstream failure count exceeds threshold, isolate and quarantine the dependency.",
		"condition": {"downstream_failure_count_gt": 3, "quarantine_active": False},
		"effect": {"decision": "require_review", "reason": "cascade_isolation_required", "required_action": "quarantine_failing_dependency"},
	},
	{
		"name": "bulkhead_overflow_sheds_load",
		"description": "Requests exceeding bulkhead capacity for a tenant are denied to prevent cross-tenant cascade.",
		"condition": {"bulkhead_capacity_exceeded": True},
		"effect": {"decision": "deny", "reason": "bulkhead_capacity_exceeded", "required_action": "shed_excess_load"},
	},
	{
		"name": "fallback_policy_required_for_degraded",
		"description": "When a dependency enters degraded state, a fallback policy must be active.",
		"condition": {"dependency_state": "degraded", "fallback_policy_active": False},
		"effect": {"decision": "require_review", "reason": "fallback_policy_required", "required_action": "activate_fallback_policy"},
	},
	# --- Provider lifecycle ---
	{
		"name": "provider_requires_owner",
		"description": "Identity providers require an accountable owner.",
		"condition": {"operation": "register_provider", "provider_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "provider_owner_required", "required_action": "assign_provider_owner"},
	},
	{
		"name": "provider_requires_metadata_evidence",
		"description": "Identity providers require validated metadata and test evidence before activation.",
		"condition": {"operation": "activate_provider", "provider_metadata_validated": False},
		"effect": {"decision": "deny", "reason": "provider_metadata_validation_required", "required_action": "validate_provider_metadata"},
	},
	{
		"name": "provider_requires_secret_reference",
		"description": "External providers require a vault or secret-manager reference.",
		"condition": {"operation": "activate_provider", "external_provider": True, "secret_reference_present": False},
		"effect": {"decision": "deny", "reason": "provider_secret_reference_required", "required_action": "attach_secret_reference"},
	},
	# --- Resource lifecycle ---
	{
		"name": "resource_requires_owner",
		"description": "Protected resources require an owner and registered scope.",
		"condition": {"operation": "register_resource", "resource_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "resource_owner_required", "required_action": "assign_resource_owner"},
	},
	{
		"name": "resource_requires_scope",
		"description": "Protected resources require at least one access scope.",
		"condition": {"operation": "register_resource", "scope_present": False},
		"effect": {"decision": "deny", "reason": "resource_scope_required", "required_action": "define_resource_scope"},
	},
	# --- Policy lifecycle ---
	{
		"name": "policy_requires_owner",
		"description": "Policies require an accountable owner.",
		"condition": {"operation": "create_policy", "policy_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "policy_owner_required", "required_action": "assign_policy_owner"},
	},
	{
		"name": "sensitive_policy_requires_conditions",
		"description": "Sensitive-resource policies require explicit conditions.",
		"condition": {"operation": "create_policy", "sensitive_resource": True, "policy_conditions_present": False},
		"effect": {"decision": "deny", "reason": "policy_conditions_required", "required_action": "define_policy_conditions"},
	},
	{
		"name": "high_risk_policy_requires_simulation",
		"description": "High-risk policy changes require simulation evidence.",
		"condition": {"operation": "activate_policy", "risk_level": "high", "simulation_evidence_present": False},
		"effect": {"decision": "require_review", "reason": "policy_simulation_required", "required_action": "attach_policy_simulation"},
	},
	# --- Grant lifecycle ---
	{
		"name": "privileged_grant_requires_approval",
		"description": "Privileged grants require approval.",
		"condition": {"operation": "create_grant", "privileged_scope": True, "approval_recorded": False},
		"effect": {"decision": "deny", "reason": "privileged_grant_approval_required", "required_action": "record_grant_approval"},
	},
	{
		"name": "privileged_grant_requires_expiry",
		"description": "Privileged grants require an expiry time.",
		"condition": {"operation": "create_grant", "privileged_scope": True, "expiry_present": False},
		"effect": {"decision": "deny", "reason": "privileged_grant_expiry_required", "required_action": "set_grant_expiry"},
	},
	{
		"name": "grant_requires_separation_of_duties",
		"description": "A requester cannot self-approve privileged access.",
		"condition": {"operation": "create_grant", "separation_of_duties_passed": False},
		"effect": {"decision": "deny", "reason": "separation_of_duties_required", "required_action": "select_independent_approver"},
	},
	{
		"name": "grant_requires_justification",
		"description": "Access grants require business justification.",
		"condition": {"operation": "create_grant", "justification_present": False},
		"effect": {"decision": "deny", "reason": "grant_justification_required", "required_action": "attach_grant_justification"},
	},
	{
		"name": "grant_exceeding_max_duration_blocked",
		"description": "Grants may not exceed the maximum permitted duration (90 days).",
		"condition": {"operation": "create_grant", "grant_duration_days_gt": 90},
		"effect": {"decision": "deny", "reason": "grant_max_duration_exceeded", "required_action": "reduce_grant_duration"},
	},
	{
		"name": "periodic_access_review_required",
		"description": "Grants without a completed periodic review past their review due date must be suspended.",
		"condition": {"operation": "evaluate_grant", "review_overdue": True, "grant_suspended": False},
		"effect": {"decision": "require_review", "reason": "periodic_access_review_due", "required_action": "complete_periodic_access_review"},
	},
	# --- Session risk ---
	{
		"name": "high_risk_session_requires_step_up",
		"description": "High-risk sessions require adaptive step-up authentication.",
		"condition": {"operation": "evaluate_session", "risk_score_gt": 74, "step_up_completed": False},
		"effect": {"decision": "deny", "reason": "adaptive_step_up_required", "required_action": "complete_step_up_authentication"},
	},
	{
		"name": "concurrent_session_limit_enforced",
		"description": "Principals may not exceed the configured concurrent session limit.",
		"condition": {"operation": "create_session", "concurrent_session_count_gt": 5},
		"effect": {"decision": "deny", "reason": "concurrent_session_limit_exceeded", "required_action": "terminate_oldest_session"},
	},
	# --- Streaming / event bus ---
	{
		"name": "decision_requires_bytewax_stream",
		"description": "Access decisions must be emitted through Bytewax.",
		"condition": {"operation": "record_decision", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_access_decision_to_bytewax"},
	},
	{
		"name": "batch_grant_requires_bytewax",
		"description": "Batch grant changes require Bytewax lifecycle coordination.",
		"condition": {"operation": "batch_grant", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_grants_to_bytewax"},
	},
	# --- Agent governance ---
	{
		"name": "access_agent_runtime_supported",
		"description": "Access-control agents must use an approved runtime.",
		"condition": {"operation": "register_access_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "access_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "access_agent_role_supported",
		"description": "Access-control agents must use an approved role.",
		"condition": {"operation": "register_access_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "access_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_agent_action_requires_human_approval",
		"description": "Privileged access actions proposed by agents require human approval.",
		"condition": {"operation": "agent_access_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# --- Service mesh integrity ---
	{
		"name": "service_mesh_identity_required",
		"description": "All intra-mesh calls from this capability must carry a verified mesh identity (mTLS or JWT).",
		"condition": {"operation": "intra_mesh_call", "mesh_identity_verified": False},
		"effect": {"decision": "deny", "reason": "mesh_identity_required", "required_action": "attach_verified_mesh_identity"},
	},
	{
		"name": "service_mesh_policy_version_pinned",
		"description": "Mesh calls must reference a pinned policy version to prevent stale-policy races.",
		"condition": {"operation": "evaluate_access", "policy_version_pinned": False},
		"effect": {"decision": "require_review", "reason": "policy_version_must_be_pinned", "required_action": "pin_policy_version_before_evaluation"},
	},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-access/dashboard", "component": "AccessDashboard", "permission": "composition_access:view", "nav_group": "Overview"},
	{"name": "providers", "path": "/composition-access/providers", "component": "AccessProviderConsole", "permission": "composition_access:admin", "nav_group": "Identity"},
	{"name": "resources", "path": "/composition-access/resources", "component": "AccessResourceRegistry", "permission": "composition_access:govern", "nav_group": "Resources"},
	{"name": "policies", "path": "/composition-access/policies", "component": "AccessPolicyStudio", "permission": "composition_access:govern", "nav_group": "Policy"},
	{"name": "grants", "path": "/composition-access/grants", "component": "AccessGrantWorkbench", "permission": "composition_access:grant", "nav_group": "Access"},
	{"name": "decisions", "path": "/composition-access/decisions", "component": "AccessDecisionExplorer", "permission": "composition_access:view", "nav_group": "Operations"},
	{"name": "sessions", "path": "/composition-access/sessions", "component": "AccessSessionMonitor", "permission": "composition_access:operate", "nav_group": "Operations"},
	{"name": "circuit_breaker", "path": "/composition-access/circuit-breaker", "component": "AccessCircuitBreakerConsole", "permission": "composition_access:operate", "nav_group": "Resilience"},
	{"name": "access_reviews", "path": "/composition-access/access-reviews", "component": "AccessReviewConsole", "permission": "composition_access:govern", "nav_group": "Governance"},
	{"name": "agents", "path": "/composition-access/agents", "component": "AccessAgentWorkbench", "permission": "composition_access:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/composition-access/audit", "component": "AccessAuditConsole", "permission": "composition_access:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/composition-access/settings", "component": "AccessSettings", "permission": "composition_access:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_access_control",
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
		"provider_console": {"icon": "key-round", "status_indicator": "provider-pill", "risk_style": "trust-band"},
		"policy_studio": {"visual": "rule-grid", "status_style": "policy-chip"},
		"grant_workbench": {"visual": "approval-queue", "status_style": "grant-chip"},
		"decision_explorer": {"visual": "decision-timeline", "status_style": "decision-chip"},
		"session_monitor": {"visual": "risk-lane", "status_style": "session-chip"},
		"circuit_breaker_console": {"visual": "breaker-gauge", "status_style": "breaker-chip"},
		"access_review_console": {"visual": "review-queue", "status_style": "review-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_access",
		"display_name": "Access Control Integration Hub",
		"version": "1.2.0",
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/composition-access/api/v1",
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
		"stream": ACCESS_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"provider_registered",
			"provider_activated",
			"provider_deactivated",
			"resource_registered",
			"policy_created",
			"policy_activated",
			"policy_simulation_completed",
			"grant_created",
			"grant_revoked",
			"grant_expired",
			"grant_review_completed",
			"session_created",
			"session_evaluated",
			"session_step_up_triggered",
			"session_terminated",
			"access_decision_recorded",
			"circuit_breaker_tripped",
			"circuit_breaker_recovered",
			"cascade_isolation_triggered",
			"access_agent_registered",
		],
		"states": ["draft", "active", "review_required", "approved", "denied", "revoked", "blocked", "quarantined"],
		"guardrails": [
			"decision_requires_bytewax_stream",
			"batch_grant_requires_bytewax",
			"privileged_agent_action_requires_human_approval",
			"circuit_breaker_trip_requires_event",
			"cross_tenant_access_blocked",
		],
	}


def event_stream_name() -> str:
	return ACCESS_EVENT_STREAM


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
