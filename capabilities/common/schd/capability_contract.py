"""Executable capability contract for APG Scheduling and Job Orchestration."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


SUPPORTED_SCHD_AGENT_RUNTIMES: list[str] = ["codex", "claude_code", "opencode", "pi"]

SUPPORTED_SCHD_AGENT_ROLES: list[str] = [
	"schedule_designer",
	"run_observer",
	"retry_advisor",
	"capacity_planner",
	"calendar_auditor",
	"worker_coordinator",
	"lifecycle_batch_reviewer",
	"scheduler_steward",
]

PRIVILEGED_SCHD_AGENT_ROLES: list[str] = [
	"retry_advisor",
	"capacity_planner",
	"worker_coordinator",
	"lifecycle_batch_reviewer",
	"scheduler_steward",
]


DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"schedules": {
		"schedule_owner_required": True,
		"timezone_required": True,
		"calendar_policy_required": True,
		"max_active_schedules": 10000,
		"manual_run_reason_required": True,
		"pause_reason_required": True,
		"resume_audit_required": True,
	},
	"jobs": {
		"job_owner_required": True,
		"job_command_required": True,
		"retry_policy_required": True,
		"critical_job_monitoring_required": True,
		"dead_letter_queue_enabled": True,
		"max_runtime_minutes": 720,
		"external_job_approval_required": True,
	},
	"job_runs": {
		"requested_by_required": True,
		"worker_pool_ready_required": True,
		"retry_requires_failed_run": True,
		"dead_letter_reason_required": True,
		"cancel_reason_required": True,
		"run_audit_required": True,
	},
	"workers": {
		"worker_pool_required": True,
		"health_check_required": True,
		"capacity_limits_required": True,
		"autoscaling_supported": True,
		"heartbeat_required": True,
		"drain_reason_required": True,
	},
	"scheduler_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_SCHD_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_SCHD_AGENT_ROLES,
	},
	"agents": {
		"first_class": True,
		"supported_runtimes": SUPPORTED_SCHD_AGENT_RUNTIMES,
		"supported_roles": SUPPORTED_SCHD_AGENT_ROLES,
		"privileged_roles": PRIVILEGED_SCHD_AGENT_ROLES,
		"require_scope": True,
		"require_owner": True,
		"require_purpose": True,
		"require_contribution_disclosure": True,
		"require_human_approval_for_privileged_roles": True,
		"adapter_contract": "aicr_provider_neutral_schd_agent_adapter",
	},
	"governance": {
		"require_tenant_context": True,
		"audit_job_runs": True,
		"external_job_approval_required": True,
		"manual_run_reason_required": True,
		"tenant_isolation_required": True,
		"batch_event_stream": "bytewax",
	},
	"observability": {
		"audit_required": True,
		"run_metrics_required": True,
		"worker_metrics_required": True,
		"schedule_metrics_required": True,
		"event_stream": "bytewax",
	},
	"streaming": {
		"engine": "bytewax",
		"lifecycle_stream": "schd.lifecycle",
		"watermark": "event_time",
		"required_processor": "bytewax",
		"required_operations": [
			"calendar_batch",
			"worker_pool_batch",
			"job_batch",
			"schedule_batch",
			"run_batch",
			"retry_batch",
			"dead_letter_batch",
			"scheduler_agent_batch",
			"audit_batch",
		],
		"topics": [
			"schd.calendars",
			"schd.worker_pools",
			"schd.jobs",
			"schd.schedules",
			"schd.runs",
			"schd.retries",
			"schd.dead_letters",
			"schd.agents",
			"schd.audit",
		],
		"broker_core_dependency_allowed": False,
	},
	"adapters": {
		"generated_app_runtime": "service.SchdService",
		"runtime_helpers": "scheduling_runtime.py",
		"api_helpers": "api.py",
		"view_models": "views.py",
		"event_stream": "bytewax",
		"workflow": "wflo",
		"message_bus": "mqeb",
		"monitoring": "moni",
		"audit_sink": "audl",
		"ai_orchestration": "aicr",
		"agent_adapter": "aicr_provider_neutral_schd_agent_adapter",
		"notifications": "ntfy",
		"cache": "cach",
		"compensation": "comp",
		"theme": "them",
	},
	"ui": {
		"enable_schedule_console": True,
		"enable_job_monitor": True,
		"enable_worker_dashboard": True,
		"enable_calendar_manager": True,
		"enable_agent_panel": True,
		"enable_lifecycle_batch_monitor": True,
		"enable_audit": True,
		"enable_analytics": True,
	},
	"theme": {
		"default_theme": "schd_scheduler_ops",
		"allow_tenant_overrides": True,
	},
}

CONFIGURATION_SCHEMA: dict[str, Any] = {
	"type": "object",
	"required": [
		"tenant_id",
		"schedules",
		"jobs",
		"job_runs",
		"workers",
		"scheduler_agents",
		"agents",
		"governance",
		"observability",
		"streaming",
		"adapters",
		"ui",
		"theme",
	],
	"properties": {key: {"type": "object"} for key in [
		"schedules",
		"jobs",
		"job_runs",
		"workers",
		"scheduler_agents",
		"agents",
		"governance",
		"observability",
		"streaming",
		"adapters",
		"ui",
		"theme",
	]} | {
		"tenant_id": {"type": "string", "minLength": 1},
	},
}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All scheduling operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "schedule_requires_owner", "description": "Schedules require an accountable owner.", "condition": {"operation": "create_schedule", "schedule_owner_assigned": False}, "effect": {"decision": "deny", "reason": "schedule_owner_required", "required_action": "assign_schedule_owner"}},
	{"name": "timezone_required", "description": "Schedules require an explicit timezone.", "condition": {"operation": "create_schedule", "timezone_present": False}, "effect": {"decision": "deny", "reason": "timezone_required", "required_action": "set_timezone"}},
	{"name": "calendar_policy_required", "description": "Schedules require a tenant calendar policy.", "condition": {"operation": "create_schedule", "calendar_policy_present": False}, "effect": {"decision": "deny", "reason": "calendar_policy_required", "required_action": "attach_calendar_policy"}},
	{"name": "worker_pool_required", "description": "Schedules require a worker pool.", "condition": {"operation": "create_schedule", "worker_pool_present": False}, "effect": {"decision": "deny", "reason": "worker_pool_required", "required_action": "attach_worker_pool"}},
	{"name": "manual_schedule_requires_reason", "description": "Manual schedules require a run reason.", "condition": {"operation": "create_schedule", "manual_trigger": True, "manual_reason_present": False}, "effect": {"decision": "deny", "reason": "manual_run_reason_required", "required_action": "record_manual_run_reason"}},
	{"name": "event_schedule_requires_policy", "description": "Event-triggered schedules require an event policy.", "condition": {"operation": "create_schedule", "event_trigger": True, "event_policy_attached": False}, "effect": {"decision": "deny", "reason": "event_policy_required", "required_action": "attach_event_policy"}},
	{"name": "job_requires_owner", "description": "Job definitions require an accountable owner.", "condition": {"operation": "define_job", "job_owner_assigned": False}, "effect": {"decision": "deny", "reason": "job_owner_required", "required_action": "assign_job_owner"}},
	{"name": "job_requires_command", "description": "Job definitions require a command or adapter target.", "condition": {"operation": "define_job", "job_command_present": False}, "effect": {"decision": "deny", "reason": "job_command_required", "required_action": "set_job_command"}},
	{"name": "job_requires_retry_policy", "description": "Job definitions require retry policy.", "condition": {"operation": "define_job", "retry_policy_attached": False}, "effect": {"decision": "deny", "reason": "retry_policy_required", "required_action": "attach_retry_policy"}},
	{"name": "critical_job_requires_monitoring", "description": "Critical jobs require monitoring.", "condition": {"job_criticality": "critical", "monitoring_attached": False}, "effect": {"decision": "deny", "reason": "critical_job_monitoring_required", "required_action": "attach_monitoring"}},
	{"name": "external_job_requires_approval", "description": "External jobs require approval.", "condition": {"external_job": True, "approval_recorded": False}, "effect": {"decision": "deny", "reason": "external_job_approval_required", "required_action": "record_external_job_approval"}},
	{"name": "long_running_job_requires_review", "description": "Long-running jobs require review.", "condition": {"expected_runtime_minutes_gt": 720, "runtime_review_recorded": False}, "effect": {"decision": "require_review", "reason": "long_running_job_review_required", "required_action": "review_job_runtime"}},
	{"name": "worker_pool_requires_queue", "description": "Worker pools require a queue.", "condition": {"operation": "register_worker_pool", "worker_queue_present": False}, "effect": {"decision": "deny", "reason": "worker_queue_required", "required_action": "set_worker_queue"}},
	{"name": "worker_pool_requires_capacity", "description": "Worker pools require positive capacity.", "condition": {"operation": "register_worker_pool", "max_concurrency_lte": 0}, "effect": {"decision": "deny", "reason": "worker_capacity_must_be_positive", "required_action": "set_worker_capacity"}},
	{"name": "worker_pool_requires_health_check", "description": "Worker pools require health-check evidence.", "condition": {"operation": "register_worker_pool", "health_check_attached": False}, "effect": {"decision": "deny", "reason": "health_check_required", "required_action": "attach_health_check"}},
	{"name": "worker_drain_requires_reason", "description": "Draining worker pools require a reason.", "condition": {"operation": "change_worker_state", "target_worker_state": "draining", "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "worker_drain_reason_required", "required_action": "record_worker_drain_reason"}},
	{"name": "run_requires_active_schedule", "description": "Job runs require an active schedule.", "condition": {"operation": "trigger_run", "schedule_active": False}, "effect": {"decision": "deny", "reason": "schedule_not_runnable", "required_action": "enable_schedule"}},
	{"name": "run_requires_ready_worker_pool", "description": "Job runs require a ready worker pool.", "condition": {"operation": "trigger_run", "worker_pool_ready": False}, "effect": {"decision": "deny", "reason": "worker_pool_not_ready", "required_action": "restore_worker_pool"}},
	{"name": "manual_run_requires_reason", "description": "Manual runs require a reason.", "condition": {"operation": "trigger_run", "manual_trigger": True, "manual_reason_present": False}, "effect": {"decision": "deny", "reason": "manual_run_reason_required", "required_action": "record_manual_run_reason"}},
	{"name": "run_requested_by_required", "description": "Job runs require a requesting actor.", "condition": {"operation": "trigger_run", "requested_by_present": False}, "effect": {"decision": "deny", "reason": "requested_by_required", "required_action": "record_requesting_actor"}},
	{"name": "run_requires_bytewax_stream", "description": "Scheduler runtime events must use Bytewax streams.", "condition": {"operation": "trigger_run", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
	{"name": "run_completion_requires_audit", "description": "Run completion requires audit evidence.", "condition": {"operation": "complete_run", "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "run_audit_required", "required_action": "record_run_audit"}},
	{"name": "run_counts_must_be_non_negative", "description": "Run metrics must be non-negative.", "condition": {"operation": "complete_run", "run_counts_valid": False}, "effect": {"decision": "deny", "reason": "run_counts_must_be_non_negative", "required_action": "correct_run_counts"}},
	{"name": "retry_requires_failed_run", "description": "Retry requires a failed or dead-lettered run.", "condition": {"operation": "retry_run", "run_retryable": False}, "effect": {"decision": "deny", "reason": "run_not_retryable", "required_action": "select_failed_run"}},
	{"name": "dead_letter_requires_reason", "description": "Dead-lettering requires a reason.", "condition": {"operation": "dead_letter_run", "dead_letter_reason_present": False}, "effect": {"decision": "deny", "reason": "dead_letter_reason_required", "required_action": "record_dead_letter_reason"}},
	{"name": "run_cancel_requires_reason", "description": "Run cancellation requires a reason.", "condition": {"operation": "cancel_run", "cancel_reason_present": False}, "effect": {"decision": "deny", "reason": "run_cancel_reason_required", "required_action": "record_cancel_reason"}},
	{"name": "schedule_pause_requires_reason", "description": "Schedule pauses require a reason.", "condition": {"operation": "pause_schedule", "pause_reason_present": False}, "effect": {"decision": "deny", "reason": "schedule_pause_reason_required", "required_action": "record_pause_reason"}},
	{"name": "schedule_disable_requires_reason", "description": "Schedule disablement requires a reason.", "condition": {"operation": "disable_schedule", "disable_reason_present": False}, "effect": {"decision": "deny", "reason": "schedule_disable_reason_required", "required_action": "record_disable_reason"}},
	{"name": "scheduler_agent_requires_id", "description": "First-class scheduler agents require stable identifiers.", "condition": {"operation": "register_scheduler_agent", "agent_id_present": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_id_required", "required_action": "assign_scheduler_agent_id"}},
	{"name": "scheduler_agent_requires_name", "description": "First-class scheduler agents require readable names.", "condition": {"operation": "register_scheduler_agent", "agent_name_present": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_name_required", "required_action": "name_scheduler_agent"}},
	{"name": "scheduler_agent_runtime_supported", "description": "First-class scheduler agents must use a configured provider-neutral runtime.", "condition": {"operation": "register_scheduler_agent", "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_runtime_not_supported", "required_action": "choose_supported_scheduler_agent_runtime"}},
	{"name": "scheduler_agent_role_supported", "description": "First-class scheduler agents must use supported scheduler-governance roles.", "condition": {"operation": "register_scheduler_agent", "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_role_not_supported", "required_action": "choose_supported_scheduler_agent_role"}},
	{"name": "scheduler_agent_requires_scope", "description": "First-class scheduler agents require calendar, worker, job, schedule, run, retry, dead-letter, or lifecycle scope.", "condition": {"operation": "register_scheduler_agent", "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_scope_required", "required_action": "set_scheduler_agent_scope"}},
	{"name": "scheduler_agent_requires_owner", "description": "First-class scheduler agents require an accountable owner.", "condition": {"operation": "register_scheduler_agent", "agent_owner_present": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_owner_required", "required_action": "assign_scheduler_agent_owner"}},
	{"name": "scheduler_agent_requires_purpose", "description": "First-class scheduler agents require a documented scheduler-governance purpose.", "condition": {"operation": "register_scheduler_agent", "agent_purpose_present": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_purpose_required", "required_action": "document_scheduler_agent_purpose"}},
	{"name": "scheduler_agent_requires_disclosure", "description": "First-class scheduler agent contributions require visible machine-contribution disclosure.", "condition": {"operation": "register_scheduler_agent", "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "scheduler_agent_disclosure_required", "required_action": "disclose_scheduler_agent"}},
	{"name": "scheduler_agent_privileged_role_requires_human_approval", "description": "Privileged scheduler-agent roles require human approval evidence.", "condition": {"operation": "register_scheduler_agent", "privileged_role": True, "human_approval_required": False}, "effect": {"decision": "require_review", "reason": "scheduler_agent_human_approval_required", "required_action": "record_scheduler_agent_human_approval"}},
	{"name": "schd_lifecycle_batch_requires_mutations", "description": "SCHD lifecycle batches must include at least one mutation.", "condition": {"operation": "validate_schd_lifecycle_batch", "mutation_count_lte": 0}, "effect": {"decision": "deny", "reason": "schd_lifecycle_batch_empty", "required_action": "include_schd_lifecycle_mutations"}},
	{"name": "schd_lifecycle_operation_supported", "description": "SCHD lifecycle batches must use configured lifecycle operations.", "condition": {"operation": "validate_schd_lifecycle_batch", "lifecycle_operation_supported": False}, "effect": {"decision": "deny", "reason": "unsupported_schd_lifecycle_operation", "required_action": "choose_supported_schd_lifecycle_operation"}},
	{"name": "bytewax_schd_lifecycle_stream_required", "description": "SCHD lifecycle batches must be routed through Bytewax.", "condition": {"operation": "validate_schd_lifecycle_batch", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_lifecycle_stream_required", "required_action": "route_schd_lifecycle_batch_to_bytewax"}},
	{"name": "schedule_state_change_requires_audit", "description": "Schedule and run state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "scheduler_audit_event_required", "required_action": "record_scheduler_audit_event"}},
	{"name": "cross_tenant_scheduler_access_denied", "description": "Scheduler records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_scheduler_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_scheduler_mutation_requires_bytewax", "description": "Batch scheduler mutations must use Bytewax event streams.", "condition": {"operation": "batch_scheduler_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}},
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/schd/dashboard", "component": "SCHDDashboard", "permission": "schd:view", "nav_group": "Overview"},
	{"name": "schedules", "path": "/schd/schedules", "component": "ScheduleConsole", "permission": "schd:schedule", "nav_group": "Schedules"},
	{"name": "jobs", "path": "/schd/jobs", "component": "JobLibrary", "permission": "schd:run_jobs", "nav_group": "Jobs"},
	{"name": "runs", "path": "/schd/runs", "component": "RunMonitor", "permission": "schd:view", "nav_group": "Runtime"},
	{"name": "workers", "path": "/schd/workers", "component": "WorkerDashboard", "permission": "schd:manage_workers", "nav_group": "Workers"},
	{"name": "calendars", "path": "/schd/calendars", "component": "CalendarManager", "permission": "schd:schedule", "nav_group": "Schedules"},
	{"name": "agents", "path": "/schd/agents", "component": "SchedulerAgentPanel", "permission": "schd:run_jobs", "nav_group": "Runtime"},
	{"name": "lifecycle", "path": "/schd/lifecycle", "component": "SCHDLifecycleBatchMonitor", "permission": "schd:admin", "nav_group": "Operations"},
	{"name": "audit", "path": "/schd/audit", "component": "SchedulerAuditTrail", "permission": "schd:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/schd/analytics", "component": "SchedulerAnalytics", "permission": "schd:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/schd/settings", "component": "SCHDSettings", "permission": "schd:admin", "nav_group": "Administration"},
]

THEME: dict[str, Any] = {
	"name": "schd_scheduler_ops",
	"tokens": {
		"color.primary": "#28536B",
		"color.accent": "#D69E2E",
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
		"schedule_calendar": {"icon": "calendar-clock", "status_indicator": "schedule-pill", "risk_style": "calendar-band"},
		"job_run_table": {"visual": "run-list", "highlight": "runtime-chip"},
		"worker_pool": {"visual": "capacity-grid", "status_style": "health-chip"},
		"retry_panel": {"visual": "retry-ladder", "status_style": "backoff-chip"},
		"agent_panel": {"visual": "agent-roster", "status_style": "scope-chip"},
		"scheduler_agent_roster": {"visual": "agent-roster", "status_style": "approval-chip"},
		"bytewax_lifecycle_panel": {"visual": "stream-batch-monitor", "status_style": "processor-chip"},
		"audit_timeline": {"visual": "event-timeline", "status_style": "scheduler-chip"},
	},
}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.schd.lifecycle",
	"state": ["calendar_policies", "worker_pools", "jobs", "schedules", "runs", "scheduler_agents"],
	"events": [
		"calendar_policy_created",
		"worker_pool_registered",
		"worker_pool_state_changed",
		"job_defined",
		"schedule_created",
		"schedule_paused",
		"schedule_resumed",
		"schedule_disabled",
		"job_run_started",
		"job_run_completed",
		"job_run_cancelled",
		"job_run_retried",
		"job_run_dead_lettered",
		"scheduler_agent_registered",
	],
	"batch_mutation_guardrail": "batch_scheduler_mutation_requires_bytewax",
	"engine": "bytewax",
	"lifecycle_stream": "schd.lifecycle",
	"watermark": "event_time",
	"required_processor": "bytewax",
	"required_operations": DEFAULT_CONFIGURATION["streaming"]["required_operations"],
	"topics": DEFAULT_CONFIGURATION["streaming"]["topics"],
	"broker_core_dependency_allowed": False,
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return the complete executable SCHD capability contract."""
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {
		"capability": "schd",
		"display_name": "Scheduling and Job Orchestration",
		"provides": ["job_scheduling", "calendar_triggers", "worker_orchestration", "retry_policies", "job_monitoring", "scheduler_agent_composition", "run_recovery", "bytewax_scheduler_lifecycle"],
		"requires": ["wflo", "mqeb", "moni", "audl", "aicr"],
		"configuration": config,
		"configuration_schema": CONFIGURATION_SCHEMA,
		"rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)},
		"ui": {
			"shell": "apg_python",
			"view_module": config["adapters"]["view_models"],
			"api_prefix": "/schd/api/v1",
			"routes": deepcopy(UI_ROUTES),
			"template_roots": ["templates/", "static/"],
			"requires_theme": True,
		},
		"theme": deepcopy(THEME),
		"agents": agent_manifest(config),
		"streaming": streaming_manifest(config),
	}


def agent_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return first-class provider-neutral scheduler-agent composition metadata."""
	config = config or DEFAULT_CONFIGURATION
	return deepcopy(config["agents"])


def streaming_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
	"""Return Bytewax lifecycle metadata for scheduler composition state."""
	config = config or DEFAULT_CONFIGURATION
	streaming = deepcopy(STREAMING)
	streaming.update(deepcopy(config["streaming"]))
	return streaming


def evaluate_capability_rules(context: dict[str, Any]) -> dict[str, Any]:
	"""Evaluate default SCHD governance rules."""
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
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
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
