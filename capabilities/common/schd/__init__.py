"""APG Scheduling and Job Orchestration (SCHD) capability registration."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .capability_contract import evaluate_capability_rules, get_capability_contract

__version__ = "1.0.0"
__capability_id__ = "schd"
__capability_name__ = "Scheduling and Job Orchestration"
__apg_dependencies__ = ["wflo", "mqeb", "moni", "audl"]

capability_metadata: dict[str, Any] = {
	"name": "schd",
	"version": __version__,
	"display_name": __capability_name__,
	"description": "Tenant-aware schedules, jobs, triggers, workers, retries, recovery, scheduler agents, calendars, and operational job monitoring",
	"category": "workflow_automation",
	"subcategory": "scheduling",
	"vendor": "Datacraft",
	"author": "APG Platform Team",
	"license": "Commercial",
	"created_at": datetime.now(timezone.utc),
	"dependencies": __apg_dependencies__,
	"provides": ["job_scheduling", "calendar_triggers", "worker_orchestration", "retry_policies", "job_monitoring", "scheduler_agents", "run_recovery"],
	"permissions": ["schd:view", "schd:schedule", "schd:run_jobs", "schd:manage_workers", "schd:audit", "schd:admin"]
}


def register_capability() -> dict[str, Any]:
	"""Register SCHD with the APG composition engine."""
	contract = get_capability_contract()
	return {
		"name": "schd",
		"aliases": ["scheduler", "job_orchestration", "scheduled_jobs"],
		"display_name": capability_metadata["display_name"],
		"description": capability_metadata["description"],
		"version": capability_metadata["version"],
		"dependencies": capability_metadata["dependencies"],
		"optional_dependencies": ["ntfy", "cach", "comp", "them"],
		"configuration": contract["configuration"],
		"configuration_schema": contract["configuration_schema"],
		"rule_engine": contract["rule_engine"],
		"capabilities": {
			"job_scheduling": "Schedule tenant-scoped jobs with cron, interval, calendar, and event triggers",
			"calendar_triggers": "Apply timezone, holiday, blackout, and business-calendar controls",
			"worker_orchestration": "Assign jobs to workers, queues, pools, and capacity lanes",
			"retry_policies": "Govern retries, dead letters, backoff, and compensation hooks",
			"scheduler_agents": "Register scoped AI scheduler assistants for design, recovery, capacity, and audit support",
			"run_recovery": "Pause, resume, cancel, retry, and dead-letter scheduler work with evidence",
			"capability_rules": "Evaluate deterministic scheduling-governance rules",
			"visual_theming": "Apply scheduler-operations theme tokens and components"
		},
		"endpoints": {
			"schedules": "/schd/api/v1/schedules",
			"jobs": "/schd/api/v1/jobs",
			"workers": "/schd/api/v1/workers",
			"calendars": "/schd/api/v1/calendars",
			"runs": "/schd/api/v1/runs",
			"agents": "/schd/api/v1/agents",
			"audit": "/schd/api/v1/audit"
		},
		"ui_components": {route["name"]: route["path"] for route in contract["ui"]["routes"]},
		"ui_manifest": contract["ui"],
		"theme": contract["theme"],
		"streaming": contract["streaming"],
		"permissions": capability_metadata["permissions"]
	}


def get_capability_info() -> dict[str, Any]:
	"""Get SCHD capability information for composition and marketplace discovery."""
	info = capability_metadata.copy()
	info["contract"] = get_capability_contract()
	return info


__all__ = ["capability_metadata", "register_capability", "get_capability_info", "get_capability_contract", "evaluate_capability_rules", "__version__", "__capability_id__", "__capability_name__", "__apg_dependencies__"]
