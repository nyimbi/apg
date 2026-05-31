"""Regression coverage for the SCHD executable capability contract."""

from capabilities.common.schd import register_capability
from capabilities.common.schd.capability_contract import evaluate_capability_rules, get_capability_contract


def test_contract_exposes_configuration_rules_ui_and_theme():
	contract = get_capability_contract("tenant-sched", {"jobs": {"max_runtime_minutes": 120}})

	assert contract["capability"] == "schd"
	assert contract["configuration"]["tenant_id"] == "tenant-sched"
	assert contract["configuration"]["jobs"]["max_runtime_minutes"] == 120
	assert contract["configuration_schema"]["required"] == ["tenant_id", "schedules", "jobs", "job_runs", "workers", "scheduler_agents", "agents", "governance", "observability", "streaming", "adapters", "ui", "theme"]
	assert len(contract["rule_engine"]["rules"]) >= 39
	assert {route["name"] for route in contract["ui"]["routes"]} >= {"dashboard", "schedules", "jobs", "runs", "workers", "calendars", "agents", "lifecycle", "audit", "analytics", "settings"}
	assert contract["ui"]["api_prefix"] == "/schd/api/v1"
	assert contract["theme"]["tokens"]["border.radius"] == "8px"
	assert "schedule_calendar" in contract["theme"]["components"]
	assert contract["theme"]["components"]["bytewax_lifecycle_panel"]["visual"] == "stream-batch-monitor"
	assert contract["agents"]["first_class"] is True
	assert "codex" in contract["agents"]["supported_runtimes"]
	assert "scheduler_steward" in contract["agents"]["privileged_roles"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["streaming"]["required_processor"] == "bytewax"
	assert contract["streaming"]["broker_core_dependency_allowed"] is False
	assert "scheduler_agent_composition" in contract["provides"]


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

	agent_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "register_scheduler_agent",
		"agent_id_present": False,
		"agent_name_present": False,
		"agent_runtime_supported": False,
		"agent_role_supported": False,
		"agent_scope_present": False,
		"agent_owner_present": False,
		"agent_purpose_present": False,
		"agent_contribution_disclosed": False,
		"privileged_role": True,
		"human_approval_required": False,
	})
	lifecycle_result = evaluate_capability_rules({
		"tenant_context_present": True,
		"operation": "validate_schd_lifecycle_batch",
		"mutation_count": 0,
		"lifecycle_operation_supported": False,
		"event_stream": "legacy_queue",
	})

	assert agent_result["decision"] == "deny"
	assert {
		"scheduler_agent_requires_id",
		"scheduler_agent_requires_name",
		"scheduler_agent_runtime_supported",
		"scheduler_agent_role_supported",
		"scheduler_agent_requires_scope",
		"scheduler_agent_requires_owner",
		"scheduler_agent_requires_purpose",
		"scheduler_agent_requires_disclosure",
		"scheduler_agent_privileged_role_requires_human_approval",
	} <= set(agent_result["matched_rules"])
	assert lifecycle_result["matched_rules"] == [
		"schd_lifecycle_batch_requires_mutations",
		"schd_lifecycle_operation_supported",
		"bytewax_schd_lifecycle_stream_required",
	]


def test_registration_includes_full_capability_contract():
	registration = register_capability()

	assert registration["name"] == "schd"
	assert registration["configuration"]["tenant_id"] == "default"
	assert registration["rule_engine"]["type"] == "deterministic"
	assert registration["ui_manifest"]["requires_theme"] is True
	assert registration["theme"]["name"] == "schd_scheduler_ops"
	assert registration["ui_components"]["schedules"] == "/schd/schedules"
	assert registration["ui_components"]["agents"] == "/schd/agents"
	assert registration["ui_components"]["lifecycle"] == "/schd/lifecycle"
	assert "wflo" in registration["dependencies"]
	assert "audl" in registration["dependencies"]
	assert "aicr" in registration["dependencies"]
	assert "schd:run_jobs" in registration["permissions"]
	assert registration["agents"]["first_class"] is True
	assert registration["streaming"]["processor"] == "bytewax"
