"""Executable capability contract for APG workflow orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_ORCHESTRATION_AGENT_ROLES = [
	"workflow_architect",
	"bpml_reviewer",
	"release_reviewer",
	"incident_reviewer",
	"compliance_reviewer",
	"optimization_reviewer",
]
SUPPORTED_TASK_TYPES = ["automated", "human", "approval", "integration", "timer", "gateway", "compensation"]
SUPPORTED_EXECUTION_STATES = ["pending", "running", "waiting", "suspended", "completed", "failed", "cancelled", "compensating"]
SUPPORTED_WORKFLOW_TRIGGER_TYPES = ["manual", "event", "schedule", "api", "webhook", "message"]
SUPPORTED_CIRCUIT_BREAKER_STATES = ["closed", "open", "half_open"]
SUPPORTED_RISK_LEVELS = ["low", "medium", "high", "critical"]
SUPPORTED_APPROVAL_STRATEGIES = ["any", "all", "majority", "unanimous", "tiered"]
SUPPORTED_COMPENSATION_MODES = ["saga", "tcc", "choreography"]
SUPPORTED_SLA_UNITS = ["minutes", "hours", "days"]
SUPPORTED_RETRY_BACKOFF_STRATEGIES = ["fixed", "linear", "exponential", "jitter"]
SUPPORTED_WORKFLOW_SCOPES = ["tenant", "cross_capability", "cross_tenant_federated"]
SUPPORTED_RELEASE_ENVIRONMENTS = ["development", "staging", "production", "dr"]
SUPPORTED_CYCLE_DETECTION_MODES = ["static", "runtime", "both"]
SUPPORTED_IDEMPOTENCY_STRATEGIES = ["key", "content_hash", "sequence_number"]

ORCHESTRATION_EVENT_STREAM = "apg.composition.orchestration.lifecycle"

PROVIDES = [
	"workflow_definition_lifecycle",
	"workflow_graph_validation",
	"workflow_execution_lifecycle",
	"human_task_coordination",
	"workflow_release_governance",
	"workflow_rule_enforcement",
	"workflow_agents",
	"cross_tenant_workflow_isolation",
	"circuit_breaker_workflow_gate",
	"cascading_failure_workflow_containment",
	"saga_compensation_coordination",
]

REQUIRES = [
	"auth",
	"audl",
	"ntfy",
	"composition_registry",
	"composition_events",
	"composition_config",
	"composition_access",
	"moni",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"workflow_definitions": {
		"owner_required": True,
		"version_required": True,
		"start_event_required": True,
		"terminal_state_required": True,
		"task_graph_required": True,
		"cycle_detection_enabled": True,
		"cross_tenant_isolation_enforced": True,
	},
	"tasks": {
		"handler_required": True,
		"human_assignee_required": True,
		"approval_policy_required": True,
		"cross_capability_contract_required": True,
		"sla_escalation_required": True,
		"timeout_required": True,
	},
	"execution": {
		"bytewax_required": True,
		"idempotency_required": True,
		"retry_limit_required": True,
		"compensation_required_for_transactions": True,
		"max_parallel_branches": 64,
		"bulkhead_isolation_enabled": True,
	},
	"releases": {
		"validation_required": True,
		"dry_run_required": True,
		"rollback_plan_required": True,
		"approval_required_for_high_risk": True,
		"blast_radius_required_for_production": True,
	},
	"circuit_breaker": {
		"enabled": True,
		"failure_threshold": 5,
		"recovery_timeout_seconds": 60,
		"half_open_probe_count": 2,
		"cascade_isolation_enabled": True,
		"per_workflow_breaker_enabled": True,
	},
	"cascading_failure": {
		"dependency_health_check_enabled": True,
		"fallback_workflow_required": True,
		"bulkhead_isolation_enabled": True,
		"max_downstream_task_failures": 3,
		"quarantine_workflow_on_cascade": True,
		"compensation_triggered_on_cascade": True,
	},
	"automation_agents": {
		"enabled": True,
		"supported_runtimes": SUPPORTED_ORCHESTRATION_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_ORCHESTRATION_AGENT_ROLES,
		"human_approval_required": True,
		"max_autonomous_scope": "recommend_validate_and_prepare",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_state_changes": True,
		"policy_attached_for_writes": True,
		"privileged_workflow_changes_reviewed": True,
		"privilege_escalation_blocked": True,
		"cross_tenant_execution_blocked": True,
	},
	"observability": {
		"event_stream": ORCHESTRATION_EVENT_STREAM,
		"stream_processor": "bytewax",
		"emit_definition_events": True,
		"emit_execution_events": True,
		"emit_release_events": True,
		"emit_agent_events": True,
		"emit_circuit_breaker_events": True,
		"emit_cascade_events": True,
		"emit_compensation_events": True,
	},
	"adapters": {
		"authorization": "adapter",
		"audit": "adapter",
		"event_stream": "bytewax",
		"notification": "adapter",
		"registry": "adapter",
		"theme": "adapter",
		"monitoring": "adapter",
	},
	"ui": {
		"enable_dashboard": True,
		"enable_definition_library": True,
		"enable_designer": True,
		"enable_execution_console": True,
		"enable_task_console": True,
		"enable_release_console": True,
		"enable_rule_center": True,
		"enable_circuit_breaker_console": True,
		"enable_compensation_console": True,
		"enable_agent_workbench": True,
		"enable_audit_console": True,
		"enable_settings": True,
	},
	"theme": {"default_theme": "composition_orchestration_control", "allow_tenant_overrides": True},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"workflow_definitions",
		"tasks",
		"execution",
		"releases",
		"circuit_breaker",
		"cascading_failure",
		"automation_agents",
		"governance",
		"observability",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {
		"tenant_id": {"type": "string", "minLength": 1},
		"workflow_definitions": {"type": "object"},
		"tasks": {"type": "object"},
		"execution": {"type": "object"},
		"releases": {"type": "object"},
		"circuit_breaker": {"type": "object"},
		"cascading_failure": {"type": "object"},
		"automation_agents": {"type": "object"},
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
		"description": "Workflow orchestration operations require tenant context.",
		"condition": {"tenant_context_present": False},
		"effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"},
	},
	# --- Write-requires-policy ---
	{
		"name": "workflow_write_requires_policy",
		"description": "Workflow definition and execution write operations require policy attachment.",
		"condition": {"operation_type": "write", "policy_attached": False},
		"effect": {"decision": "deny", "reason": "operation_policy_required", "required_action": "attach_operation_policy"},
	},
	# --- Cross-tenant isolation ---
	{
		"name": "cross_tenant_execution_blocked",
		"description": "Workflow executions may not cross tenant boundaries without explicit federation.",
		"condition": {"cross_tenant_execution_attempted": True, "federation_approved": False},
		"effect": {"decision": "deny", "reason": "cross_tenant_execution_forbidden", "required_action": "request_workflow_federation_approval"},
	},
	{
		"name": "cross_tenant_task_delegation_blocked",
		"description": "Human tasks may not be delegated to principals in a different tenant.",
		"condition": {"operation": "delegate_task", "cross_tenant_delegation": True},
		"effect": {"decision": "deny", "reason": "cross_tenant_task_delegation_forbidden", "required_action": "select_same_tenant_delegate"},
	},
	# --- Privilege escalation prevention ---
	{
		"name": "workflow_privilege_escalation_blocked",
		"description": "A principal may not approve a workflow action that exceeds their authorised scope.",
		"condition": {"operation": "approve_workflow_action", "privilege_escalation_detected": True},
		"effect": {"decision": "deny", "reason": "workflow_privilege_escalation_forbidden", "required_action": "escalate_via_approval_chain"},
	},
	# --- Circuit breaker rules ---
	{
		"name": "circuit_breaker_open_blocks_execution",
		"description": "When the workflow circuit breaker is open, new executions are denied.",
		"condition": {"circuit_breaker_state": "open", "operation": "start_execution"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_open", "required_action": "wait_for_circuit_recovery"},
	},
	{
		"name": "circuit_breaker_half_open_limits_executions",
		"description": "In half-open state only probe executions are permitted.",
		"condition": {"circuit_breaker_state": "half_open", "probe_budget_exhausted": True},
		"effect": {"decision": "deny", "reason": "circuit_breaker_half_open_budget_exhausted", "required_action": "queue_execution_until_probe_completes"},
	},
	{
		"name": "circuit_breaker_trip_requires_event",
		"description": "Circuit breaker state transitions must emit a Bytewax lifecycle event.",
		"condition": {"operation": "trip_circuit_breaker", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "circuit_breaker_event_required", "required_action": "emit_circuit_breaker_event_to_bytewax"},
	},
	# --- Cascading failure containment ---
	{
		"name": "cascade_isolation_on_task_failure",
		"description": "When downstream task failures exceed threshold, quarantine the workflow branch.",
		"condition": {"downstream_failure_count_gt": 3, "workflow_quarantine_active": False},
		"effect": {"decision": "require_review", "reason": "workflow_cascade_isolation_required", "required_action": "quarantine_workflow_branch"},
	},
	{
		"name": "bulkhead_overflow_queues_executions",
		"description": "New executions exceeding per-tenant bulkhead limit are queued, not denied.",
		"condition": {"operation": "start_execution", "bulkhead_capacity_exceeded": True, "queue_available": False},
		"effect": {"decision": "deny", "reason": "bulkhead_capacity_exceeded_and_queue_full", "required_action": "reject_execution_until_capacity_available"},
	},
	{
		"name": "compensation_triggered_on_transactional_cascade",
		"description": "When a transactional workflow enters cascade-failure state, compensation is triggered automatically.",
		"condition": {"transactional_workflow": True, "cascade_failure_detected": True, "compensation_triggered": False},
		"effect": {"decision": "require_review", "reason": "compensation_trigger_required", "required_action": "trigger_saga_compensation"},
	},
	# --- Workflow definition lifecycle ---
	{
		"name": "workflow_requires_owner",
		"description": "Workflow definitions require an accountable owner.",
		"condition": {"operation": "define_workflow", "workflow_owner_assigned": False},
		"effect": {"decision": "deny", "reason": "workflow_owner_required", "required_action": "assign_workflow_owner"},
	},
	{
		"name": "workflow_requires_version",
		"description": "Workflow definitions require a version.",
		"condition": {"operation": "define_workflow", "workflow_version_present": False},
		"effect": {"decision": "deny", "reason": "workflow_version_required", "required_action": "set_workflow_version"},
	},
	{
		"name": "workflow_requires_start_event",
		"description": "Workflow definitions require at least one start event.",
		"condition": {"operation": "define_workflow", "start_event_present": False},
		"effect": {"decision": "deny", "reason": "workflow_start_event_required", "required_action": "add_start_event"},
	},
	{
		"name": "workflow_requires_task_graph",
		"description": "Workflow definitions require at least one executable task.",
		"condition": {"operation": "define_workflow", "task_graph_present": False},
		"effect": {"decision": "deny", "reason": "workflow_task_graph_required", "required_action": "add_executable_task_graph"},
	},
	{
		"name": "workflow_requires_terminal_state",
		"description": "Workflow definitions require a terminal state.",
		"condition": {"operation": "define_workflow", "terminal_state_present": False},
		"effect": {"decision": "deny", "reason": "workflow_terminal_state_required", "required_action": "add_terminal_state"},
	},
	{
		"name": "compensation_required_for_transaction",
		"description": "Transactional workflows require compensation steps.",
		"condition": {"operation": "define_workflow", "transactional_workflow": True, "compensation_present": False},
		"effect": {"decision": "deny", "reason": "compensation_required", "required_action": "attach_compensation_steps"},
	},
	# --- Task lifecycle ---
	{
		"name": "task_requires_handler",
		"description": "Automated and integration tasks require a handler.",
		"condition": {"operation": "define_task", "handler_present": False},
		"effect": {"decision": "deny", "reason": "task_handler_required", "required_action": "attach_task_handler"},
	},
	{
		"name": "task_requires_timeout",
		"description": "All tasks must declare a timeout to prevent unbounded blocking.",
		"condition": {"operation": "define_task", "timeout_configured": False},
		"effect": {"decision": "deny", "reason": "task_timeout_required", "required_action": "configure_task_timeout"},
	},
	{
		"name": "human_task_requires_assignee",
		"description": "Human tasks require an assignee, group, or role.",
		"condition": {"operation": "define_task", "human_task": True, "assignee_present": False},
		"effect": {"decision": "deny", "reason": "human_task_assignee_required", "required_action": "assign_human_task"},
	},
	{
		"name": "approval_task_requires_policy",
		"description": "Approval tasks require an approval policy.",
		"condition": {"operation": "define_task", "approval_task": True, "approval_policy_present": False},
		"effect": {"decision": "deny", "reason": "approval_policy_required", "required_action": "attach_approval_policy"},
	},
	{
		"name": "cross_capability_task_requires_contract",
		"description": "Cross-capability tasks require an APG capability contract reference.",
		"condition": {"operation": "define_task", "cross_capability_task": True, "capability_contract_present": False},
		"effect": {"decision": "deny", "reason": "capability_contract_required", "required_action": "attach_capability_contract"},
	},
	{
		"name": "retry_policy_requires_limit",
		"description": "Retry policies require bounded attempts.",
		"condition": {"operation": "define_task", "retry_policy_present": True, "retry_limit_present": False},
		"effect": {"decision": "deny", "reason": "retry_limit_required", "required_action": "set_retry_limit"},
	},
	{
		"name": "sla_task_requires_escalation",
		"description": "SLA-bound tasks require escalation rules.",
		"condition": {"operation": "define_task", "sla_present": True, "escalation_present": False},
		"effect": {"decision": "require_review", "reason": "sla_escalation_required", "required_action": "attach_escalation_rule"},
	},
	# --- Execution lifecycle ---
	{
		"name": "execution_requires_bytewax_stream",
		"description": "Workflow execution lifecycle events must use Bytewax.",
		"condition": {"operation": "start_execution", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_execution_lifecycle_to_bytewax"},
	},
	{
		"name": "execution_requires_idempotency_key",
		"description": "Workflow execution starts require idempotency keys.",
		"condition": {"operation": "start_execution", "idempotency_key_present": False},
		"effect": {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"},
	},
	{
		"name": "high_risk_execution_requires_review",
		"description": "High-risk workflow executions require review.",
		"condition": {"operation": "start_execution", "risk_level": "high", "review_recorded": False},
		"effect": {"decision": "require_review", "reason": "high_risk_review_required", "required_action": "record_execution_review"},
	},
	# --- Release lifecycle ---
	{
		"name": "release_requires_validation",
		"description": "Workflow releases require validation evidence.",
		"condition": {"operation": "release_workflow", "validation_evidence_present": False},
		"effect": {"decision": "deny", "reason": "release_validation_required", "required_action": "attach_validation_evidence"},
	},
	{
		"name": "release_requires_dry_run",
		"description": "Workflow releases require a dry-run result.",
		"condition": {"operation": "release_workflow", "dry_run_passed": False},
		"effect": {"decision": "deny", "reason": "release_dry_run_required", "required_action": "run_release_dry_run"},
	},
	{
		"name": "release_requires_rollback_plan",
		"description": "Workflow releases require a rollback plan.",
		"condition": {"operation": "release_workflow", "rollback_plan_present": False},
		"effect": {"decision": "deny", "reason": "release_rollback_plan_required", "required_action": "attach_rollback_plan"},
	},
	{
		"name": "production_release_requires_blast_radius",
		"description": "Production workflow releases require a blast-radius estimate.",
		"condition": {"operation": "release_workflow", "environment": "production", "blast_radius_estimated": False},
		"effect": {"decision": "deny", "reason": "blast_radius_estimation_required", "required_action": "attach_blast_radius_estimate"},
	},
	# --- Batch / streaming ---
	{
		"name": "batch_schedule_requires_bytewax",
		"description": "Batch execution scheduling requires Bytewax coordination.",
		"condition": {"operation": "batch_schedule", "event_stream_ne": "bytewax"},
		"effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "route_batch_schedule_to_bytewax"},
	},
	# --- Agent governance ---
	{
		"name": "workflow_agent_runtime_supported",
		"description": "Workflow agents must use an approved runtime.",
		"condition": {"operation": "register_workflow_agent", "agent_runtime_supported": False},
		"effect": {"decision": "deny", "reason": "workflow_agent_runtime_not_supported", "required_action": "select_supported_agent_runtime"},
	},
	{
		"name": "workflow_agent_role_supported",
		"description": "Workflow agents must use an approved role.",
		"condition": {"operation": "register_workflow_agent", "agent_role_supported": False},
		"effect": {"decision": "deny", "reason": "workflow_agent_role_not_supported", "required_action": "select_supported_agent_role"},
	},
	{
		"name": "privileged_agent_workflow_action_requires_human_approval",
		"description": "Privileged workflow actions proposed by agents require human approval.",
		"condition": {"operation": "agent_workflow_action", "privileged_scope": True, "human_approval_recorded": False},
		"effect": {"decision": "deny", "reason": "human_approval_required", "required_action": "record_human_approval"},
	},
	# --- Service mesh integrity ---
	{
		"name": "intra_mesh_workflow_call_requires_identity",
		"description": "Cross-capability workflow task invocations must carry a verified mesh identity.",
		"condition": {"operation": "cross_capability_task_invoke", "mesh_identity_verified": False},
		"effect": {"decision": "deny", "reason": "mesh_identity_required", "required_action": "attach_verified_mesh_identity"},
	},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/composition-orchestration/dashboard", "component": "OrchestrationDashboard", "permission": "composition_orchestration:view", "nav_group": "Overview"},
	{"name": "definitions", "path": "/composition-orchestration/definitions", "component": "WorkflowDefinitionLibrary", "permission": "composition_orchestration:manage_definitions", "nav_group": "Definitions"},
	{"name": "designer", "path": "/composition-orchestration/designer", "component": "WorkflowDesigner", "permission": "composition_orchestration:design", "nav_group": "Design"},
	{"name": "executions", "path": "/composition-orchestration/executions", "component": "WorkflowExecutionConsole", "permission": "composition_orchestration:operate", "nav_group": "Operations"},
	{"name": "tasks", "path": "/composition-orchestration/tasks", "component": "WorkflowTaskConsole", "permission": "composition_orchestration:manage_tasks", "nav_group": "Operations"},
	{"name": "releases", "path": "/composition-orchestration/releases", "component": "WorkflowReleaseConsole", "permission": "composition_orchestration:release", "nav_group": "Release"},
	{"name": "rules", "path": "/composition-orchestration/rules", "component": "WorkflowRuleCenter", "permission": "composition_orchestration:govern", "nav_group": "Governance"},
	{"name": "circuit_breaker", "path": "/composition-orchestration/circuit-breaker", "component": "WorkflowCircuitBreakerConsole", "permission": "composition_orchestration:operate", "nav_group": "Resilience"},
	{"name": "compensation", "path": "/composition-orchestration/compensation", "component": "WorkflowCompensationConsole", "permission": "composition_orchestration:operate", "nav_group": "Resilience"},
	{"name": "agents", "path": "/composition-orchestration/agents", "component": "WorkflowAgentWorkbench", "permission": "composition_orchestration:admin", "nav_group": "Automation"},
	{"name": "audit", "path": "/composition-orchestration/audit", "component": "WorkflowAuditConsole", "permission": "composition_orchestration:audit", "nav_group": "Governance"},
	{"name": "settings", "path": "/composition-orchestration/settings", "component": "WorkflowSettings", "permission": "composition_orchestration:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "composition_orchestration_control",
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
		"definition_library": {"icon": "workflow", "status_indicator": "definition-pill", "risk_style": "validation-band"},
		"designer": {"visual": "graph-canvas", "status_style": "graph-chip"},
		"execution_console": {"visual": "run-lanes", "status_style": "execution-chip"},
		"task_console": {"visual": "task-queue", "status_style": "sla-chip"},
		"release_console": {"visual": "release-checklist", "status_style": "evidence-chip"},
		"rule_center": {"visual": "rule-grid", "status_style": "guardrail-chip"},
		"circuit_breaker_console": {"visual": "breaker-gauge", "status_style": "breaker-chip"},
		"compensation_console": {"visual": "saga-timeline", "status_style": "compensation-chip"},
		"agent_workbench": {"visual": "review-lane", "status_style": "approval-chip"},
	},
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "composition_orchestration",
		"display_name": "Workflow Orchestration",
		"version": "1.2.0",
		"provides": deepcopy(PROVIDES),
		"requires": deepcopy(REQUIRES),
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "default_decision": "allow", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": "views.py",
			"api_prefix": "/composition-orchestration/api/v1",
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
		"stream": ORCHESTRATION_EVENT_STREAM,
		"key": "tenant_id",
		"events": [
			"workflow_defined",
			"workflow_validated",
			"workflow_released",
			"workflow_execution_started",
			"workflow_execution_advanced",
			"workflow_execution_completed",
			"workflow_execution_failed",
			"workflow_execution_compensating",
			"workflow_task_assigned",
			"workflow_task_completed",
			"workflow_task_escalated",
			"circuit_breaker_tripped",
			"circuit_breaker_recovered",
			"cascade_isolation_triggered",
			"compensation_saga_started",
			"compensation_saga_completed",
			"workflow_agent_registered",
		],
		"states": ["draft", "validated", "released", "running", "waiting", "compensating", "completed", "failed", "quarantined", "retired"],
		"guardrails": [
			"execution_requires_bytewax_stream",
			"batch_schedule_requires_bytewax",
			"circuit_breaker_trip_requires_event",
			"privileged_agent_workflow_action_requires_human_approval",
			"cross_tenant_execution_blocked",
		],
	}


def event_stream_name() -> str:
	return ORCHESTRATION_EVENT_STREAM


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
