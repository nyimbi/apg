"""Executable capability contract for APG Distributed Computing."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIGURATION: dict[str, Any] = {
	"tenant_id": "default",
	"jobs": {"job_owner_required": True, "idempotency_key_required": True, "retry_policy_required": True, "max_partitions": 1000},
	"workers": {"worker_pool_required": True, "health_check_required": True, "capacity_quota_required": True, "autoscaling_supported": True},
	"coordination": {"event_bus_required": True, "distributed_locking_enabled": True, "result_aggregation_required": True, "dead_letter_queue_enabled": True},
	"governance": {"require_tenant_context": True, "audit_job_execution": True, "quota_policy_required": True, "monitoring_required": True},
	"ui": {"enable_compute_dashboard": True, "enable_job_console": True, "enable_worker_pool": True, "enable_partition_monitor": True},
	"theme": {"default_theme": "dist_compute_grid", "allow_tenant_overrides": True}
}

CONFIGURATION_SCHEMA: dict[str, Any] = {"type": "object", "required": ["tenant_id", "jobs", "workers", "coordination", "governance", "ui", "theme"], "properties": {key: {"type": "object"} for key in ["jobs", "workers", "coordination", "governance", "ui", "theme"]} | {"tenant_id": {"type": "string", "minLength": 1}}}

RULES: list[dict[str, Any]] = [
	{"name": "tenant_context_required", "description": "All distributed operations require tenant context.", "condition": {"tenant_context_present": False}, "effect": {"decision": "deny", "reason": "tenant_context_required", "required_action": "attach_tenant_context"}},
	{"name": "job_requires_owner", "description": "Distributed jobs require an accountable owner.", "condition": {"operation": "submit_job", "job_owner_assigned": False}, "effect": {"decision": "deny", "reason": "job_owner_required", "required_action": "assign_job_owner"}},
	{"name": "idempotency_key_required", "description": "Distributed jobs require idempotency keys.", "condition": {"operation": "submit_job", "idempotency_key_present": False}, "effect": {"decision": "deny", "reason": "idempotency_key_required", "required_action": "attach_idempotency_key"}},
	{"name": "worker_pool_requires_health", "description": "Worker pools require health checks.", "condition": {"worker_pool_selected": True, "health_check_attached": False}, "effect": {"decision": "deny", "reason": "worker_health_required", "required_action": "attach_worker_health_check"}},
	{"name": "quota_policy_required", "description": "Distributed execution requires tenant quota policy.", "condition": {"quota_policy_attached": False, "job_submission_requested": True}, "effect": {"decision": "deny", "reason": "quota_policy_required", "required_action": "attach_quota_policy"}},
	{"name": "large_partition_job_requires_review", "description": "Large partitioned jobs require review.", "condition": {"partition_count_gt": 1000, "partition_review_recorded": False}, "effect": {"decision": "require_review", "reason": "partition_review_required", "required_action": "review_partition_plan"}}
]

UI_ROUTES: list[dict[str, str]] = [
	{"name": "dashboard", "path": "/dist/dashboard", "component": "DISTDashboard", "permission": "dist:view", "nav_group": "Overview"},
	{"name": "jobs", "path": "/dist/jobs", "component": "DistributedJobs", "permission": "dist:submit_jobs", "nav_group": "Jobs"},
	{"name": "workers", "path": "/dist/workers", "component": "WorkerPools", "permission": "dist:manage_workers", "nav_group": "Workers"},
	{"name": "partitions", "path": "/dist/partitions", "component": "PartitionMonitor", "permission": "dist:view", "nav_group": "Runtime"},
	{"name": "queues", "path": "/dist/queues", "component": "QueueMonitor", "permission": "dist:view", "nav_group": "Runtime"},
	{"name": "scaling", "path": "/dist/scaling", "component": "ScalingPolicy", "permission": "dist:scale", "nav_group": "Operations"},
	{"name": "analytics", "path": "/dist/analytics", "component": "ComputeAnalytics", "permission": "dist:view", "nav_group": "Operations"},
	{"name": "settings", "path": "/dist/settings", "component": "DISTSettings", "permission": "dist:admin", "nav_group": "Administration"}
]

THEME: dict[str, Any] = {"name": "dist_compute_grid", "tokens": {"color.primary": "#2A4365", "color.accent": "#38A169", "color.success": "#2F855A", "color.warning": "#B7791F", "color.danger": "#C53030", "surface.canvas": "#F7F8FA", "surface.panel": "#FFFFFF", "text.primary": "#172033", "text.secondary": "#52606D", "border.radius": "8px", "density": "compact"}, "components": {"job_grid": {"icon": "network", "status_indicator": "job-pill", "risk_style": "quota-band"}, "worker_pool": {"visual": "capacity-grid", "highlight": "health-chip"}, "partition_monitor": {"visual": "shard-map", "status_style": "partition-chip"}, "scaling_panel": {"visual": "scale-graph", "status_style": "autoscale-chip"}}}


def get_capability_contract(tenant_id: str = "default", overrides: dict[str, Any] | None = None) -> dict[str, Any]:
	config = deepcopy(DEFAULT_CONFIGURATION)
	config["tenant_id"] = tenant_id
	if overrides:
		_deep_merge(config, overrides)
	return {"capability": "dist", "display_name": "Distributed Computing", "configuration": config, "configuration_schema": CONFIGURATION_SCHEMA, "rule_engine": {"type": "deterministic", "rules": deepcopy(RULES)}, "ui": {"shell": "apg_python", "view_module": "views.py", "api_prefix": "/dist/api/v1", "routes": deepcopy(UI_ROUTES), "template_roots": ["templates/", "static/"], "requires_theme": True}, "theme": deepcopy(THEME)}


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
		if key.endswith("_lt"):
			if not context.get(key[:-3], 0) < expected:
				return False
		elif key.endswith("_gt"):
			if not context.get(key[:-3], 0) > expected:
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
