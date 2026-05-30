"""Regression coverage for the SCHD executable capability contract."""

from capabilities.common.schd import register_capability
from capabilities.common.schd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sched", {"jobs": {"max_runtime_minutes": 120}})

	assert contract["capability"] == "schd"
	assert contract["configuration"]["tenant_id"] == "tenant-sched"
	assert contract["configuration"]["jobs"]["max_runtime_minutes"] == 120
	assert contract["configuration_schema"]["required"] == ["tenant_id", "schedules", "jobs", "job_runs", "workers", "scheduler_agents", "governance", "observability", "adapters", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 30
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "schedules", "jobs", "runs", "workers", "calendars", "agents", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/schd/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "schedule_calendar" in contract["theme"]["components"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert "scheduler_agents" in contract["provides"]


def test_rule_engine_enforces_scheduler_guardrails():
	result = evaluate_capability_rules({
		"tenant_context_present": False,
		"operation": "create_schedule",
		"schedule_owner_assigned": False,
		"timezone_present": False,
		"job_criticality": "critical",
		"monitoring_attached": False,
		"external_job": True,
		"approval_recorded": False,
		"expected_runtime_minutes": 900,
		"runtime_review_recorded": False,
		"calendar_policy_present": False,
		"worker_pool_present": False,
	})

	assert result["decision"] == "deny"
	assert set(result["matched_rules"]) >= {"tenant_context_required", "schedule_requires_owner", "timezone_required", "calendar_policy_required", "worker_pool_required", "critical_job_requires_monitoring", "external_job_requires_approval", "long_running_job_requires_review"}


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "schd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "schd_scheduler_ops"
	assert registration["ui_components"]["schedules"] == "/schd/schedules"
	assert registration["ui_components"]["agents"] == "/schd/agents"
	assert "wflo" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert "schd:run_jobs" in registration["permissions"]
	assert registration["streaming"]["processor"] == "bytewax"
