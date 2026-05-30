"""Executable capability contract for APG Distributed Computing."""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any

SUPPORTED_COMPUTE_AGENT_RUNTIMES = ["codex", "claude_code", "opencode", "pi"]
SUPPORTED_COMPUTE_AGENT_ROLES = ["job_planner", "partition_operator", "worker_pool_operator", "result_reviewer", "incident_reviewer"]

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"jobs": {"job_owner_required": True, "idempotency_key_required": True, "retry_policy_required": True, "max_partitions": 1000},
	"workers": {"worker_pool_required": True, "health_check_required": True, "capacity_quota_required": True, "autoscaling_supported": True},
	"coordination": {"event_stream": "bytewax", "event_stream_required": True, "distributed_locking_enabled": True, "result_aggregation_required": True, "dead_letter_queue_enabled": True},
	"compute_agents": {
		"agent_assist_enabled": True,
		"agent_registration_required": True,
		"agent_runtime_required": True,
		"agent_scope_required": True,
		"agent_contribution_disclosure_required": True,
		"supported_runtimes": SUPPORTED_COMPUTE_AGENT_RUNTIMES,
		"allowed_roles": SUPPORTED_COMPUTE_AGENT_ROLES,
	},
	"governance": {"require_tenant_context": True, "audit_job_execution": True, "quota_policy_required": True, "monitoring_required": True, "tenant_isolation_required": True, "state_change_reason_required": True, "batch_event_stream": "bytewax"},
	"observability": {"audit_required": True, "trace_required": True, "queue_metrics_required": True, "agent_activity_required": True, "event_stream": "bytewax"},
	"adapters": {"generated_app_runtime": "service.DistService", "api_helpers": "api.py", "view_models": "views.py", "event_stream": "bytewax", "scheduler": "schd", "monitoring": "moni", "configuration": "conf", "logs": "logt", "cache": "cach", "edge": "edge", "audit_sink": "audl"},
	"ui": {"enable_compute_dashboard": True, "enable_job_console": True, "enable_worker_pool": True, "enable_partition_monitor": True, "enable_agent_panel": True, "enable_audit": True, "enable_analytics": True},
	"theme": {"default_theme": "dist_compute_grid", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "jobs", "workers", "coordination", "compute_agents", "governance", "observability", "adapters", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["jobs", "workers", "coordination", "compute_agents", "governance", "observability", "adapters", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All distributed operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "job_requires_owner", "description": "Distributed jobs require an accountable owner.", "condition": {"operation": "submit_job", "job_owner_assigned": False}, "effect": {"decision": "deny", "reason": "job_owner_required", "required_action": "assign_job_owner"}},
	{"name": "idempotency_key_required", "description": "Distributed jobs require idempotency keys.", "condition": {"operation": "submit_job", "idempotency_key_present": False}, "effect": {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"}},
	{"name": "retry_policy_required", "description": "Distributed jobs require retry policy.", "condition": {"operation": "submit_job", "retry_policy_attached": False}, "effect": {"decision": "deny", "reason": "retry_policy_required", "required_action": "attach_retry_policy"}},
	{"name": "event_stream_required", "description": "Distributed jobs require an event stream.", "condition": {"operation": "submit_job", "event_stream_attached": False}, "effect": {"decision": "deny", "reason": "event_stream_required", "required_action": "attach_event_stream"}},
	{"name": "result_aggregation_required", "description": "Distributed jobs require result aggregation strategy.", "condition": {"operation": "submit_job", "aggregation_strategy_attached": False}, "effect": {"decision": "deny", "reason": "result_aggregation_required", "required_action": "attach_aggregation_strategy"}},
	{"name": "worker_pool_requires_health", "description": "Worker pools require health checks.", "condition": {"worker_pool_selected": True, "health_check_attached": False}, "effect": {"decision": "deny", "reason": "worker_health_required", "required_action": "attach_worker_health_check"}},
	{"name": "partition_count_required", "description": "Distributed jobs require at least one partition.", "condition": {"operation": "submit_job", "partition_count_lte": 0}, "effect": {"decision": "deny", "reason": "partition_count_required", "required_action": "set_partition_count"}},
	{"name": "quota_policy_required", "description": "Distributed execution requires tenant quota policy.", "condition": {"quota_policy_attached": False, "job_submission_requested": True}, "effect": {"decision": "deny", "reason": "quota_policy_required", "required_action": "attach_quota_policy"}},
	{"name": "large_partition_job_requires_review", "description": "Large partitioned jobs require review.", "condition": {"partition_count_gt": 1000, "partition_review_recorded": False}, "effect": {"decision": "require_review", "reason": "partition_review_required", "required_action": "review_partition_plan"}},
	{"name": "compute_agent_requires_registration", "description": "AI compute agents must be registered.", "condition": {"compute_agent_present": True, "agent_registered": False}, "effect": {"decision": "deny", "reason": "compute_agent_registration_required", "required_action": "register_compute_agent"}},
	{"name": "compute_agent_runtime_supported", "description": "AI compute agents must use a supported runtime.", "condition": {"compute_agent_present": True, "agent_runtime_supported": False}, "effect": {"decision": "deny", "reason": "compute_agent_runtime_not_supported", "required_action": "choose_supported_compute_agent_runtime"}},
	{"name": "compute_agent_role_supported", "description": "AI compute agents must use a supported role.", "condition": {"compute_agent_present": True, "agent_role_supported": False}, "effect": {"decision": "deny", "reason": "compute_agent_role_not_supported", "required_action": "choose_supported_compute_agent_role"}},
	{"name": "compute_agent_requires_scope", "description": "AI compute agents require explicit scope.", "condition": {"compute_agent_present": True, "agent_scope_present": False}, "effect": {"decision": "deny", "reason": "compute_agent_scope_required", "required_action": "set_compute_agent_scope"}},
	{"name": "compute_agent_requires_disclosure", "description": "AI compute-agent contributions require disclosure.", "condition": {"compute_agent_present": True, "agent_contribution_disclosed": False}, "effect": {"decision": "deny", "reason": "compute_agent_disclosure_required", "required_action": "disclose_compute_agent"}},
	{"name": "dist_state_change_requires_reason", "description": "Distributed job state changes require a reason.", "condition": {"state_change_requested": True, "state_change_reason_present": False}, "effect": {"decision": "deny", "reason": "dist_state_change_reason_required", "required_action": "record_state_change_reason"}},
	{"name": "dist_state_change_requires_audit", "description": "Distributed job state changes require audit evidence.", "condition": {"state_change_requested": True, "audit_event_recorded": False}, "effect": {"decision": "deny", "reason": "dist_audit_event_required", "required_action": "record_dist_audit_event"}},
	{"name": "cross_tenant_compute_access_denied", "description": "Distributed-compute records may not cross tenant boundaries.", "condition": {"cross_tenant_access": True}, "effect": {"decision": "deny", "reason": "cross_tenant_compute_access_denied", "required_action": "use_tenant_local_context"}},
	{"name": "batch_compute_mutation_requires_bytewax", "description": "Batch compute mutations must use Bytewax event streams.", "condition": {"operation": "batch_compute_mutation", "event_stream_ne": "bytewax"}, "effect": {"decision": "deny", "reason": "bytewax_event_stream_required", "required_action": "use_bytewax_event_stream"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dist/dashboard", "component": "DISTDashboard", "permission": "dist:view", "nav_group": "Overview"},
	{"name": "jobs", "path": "/dist/jobs", "component": "DistributedJobs", "permission": "dist:submit_jobs", "nav_group": "Jobs"},
	{"name": "workers", "path": "/dist/workers", "component": "WorkerPools", "permission": "dist:manage_workers", "nav_group": "Workers"},
	{"name": "partitions", "path": "/dist/partitions", "component": "PartitionMonitor", "permission": "dist:view", "nav_group": "Runtime"},
	{"name": "queues", "path": "/dist/queues", "component": "QueueMonitor", "permission": "dist:view", "nav_group": "Runtime"},
	{"name": "scaling", "path": "/dist/scaling", "component": "ScalingPolicy", "permission": "dist:scale", "nav_group": "Operations"},
	{"name": "agents", "path": "/dist/agents", "component": "ComputeAgentPanel", "permission": "dist:submit_jobs", "nav_group": "Agents"},
	{"name": "audit", "path": "/dist/audit", "component": "ComputeAuditTrail", "permission": "dist:audit", "nav_group": "Governance"},
	{"name": "analytics", "path": "/dist/analytics", "component": "ComputeAnalytics", "permission": "dist:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/dist/settings", "component": "DISTSettings", "permission": "dist:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "dist_compute_grid", "tokens": {"color.primary": "#2A4365", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"job_grid": {"icon": "network", "status_indicator": "job-pill", "risk_style": "quota-band"}, "worker_pool": {"visual": "capacity-grid", "highlight": "health-chip"}, "partition_monitor": {"visual": "shard-map", "status_style": "partition-chip"}, "scaling_panel": {"visual": "scale-graph", "status_style": "autoscale-chip"}, "agent_panel": {"icon": "bot", "status_style": "scope-chip"}, "audit_timeline": {"icon": "list-checks", "status_style": "governance-chip"}}}

STREAMING: dict[str, Any] = {
	"processor": "bytewax",
	"topic": "apg.dist.lifecycle",
	"state": ["worker_pools", "workers", "jobs", "partitions", "aggregations", "scaling_decisions", "compute_agents", "audit_events"],
	"events": ["worker_pool_created", "worker_registered", "job_submitted", "partition_review_approved", "job_state_changed", "partitions_dispatched", "partition_completed", "partition_failed", "results_aggregated", "scaling_decision_recorded", "compute_agent_registered"],
	"batch_mutation_guardrail": "batch_compute_mutation_requires_bytewax",
}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "dist", "display_name": "Distributed Computing", "provides": ["distributed_jobs", "worker_pools", "partitioned_execution", "coordination", "distributed_scaling", "compute_agents"], "requires": ["mqeb", "moni", "conf"], "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": config["adapters"]["view_models"], "api_prefix": "/dist/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME), "streaming": deepcopy(STREAMING)}


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
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual <= expected:
				return False
		elif key.endswith("_lt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual < expected:
				return False
		elif key.endswith("_gt"):
			actual = context.get(key[:-3])
			if not isinstance(actual, Number) or not actual > expected:
				return False
		elif key.endswith("_gte"):
			actual = context.get(key[:-4])
			if not isinstance(actual, Number) or not actual >= expected:
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
