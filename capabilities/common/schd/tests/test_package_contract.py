"""SCHD package runtime and publish contract tests."""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest

from capabilities.capability_contract_registry import validate_contract_shape
from capabilities.common.schd.service import SchdService


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
	spec = importlib.util.spec_from_file_location(name, path)
	assert spec is not None
	assert spec.loader is not None
	module = importlib.util.module_from_spec(spec)
	sys.modules[name] = module
	spec.loader.exec_module(module)
	return module


def test_schd_contract_shape_is_valid():
	module = _load_module("schd_contract", PACKAGE_DIR / "capability_contract.py")
	contract = module.get_capability_contract("tenant-schedule")

	validate_contract_shape(contract, PACKAGE_DIR / "capability_contract.py")
	assert contract["capability"] == "schd"
	assert contract["configuration"]["jobs"]["retry_policy_required"] is True
	assert contract["configuration"]["workers"]["capacity_limits_required"] is True
	assert contract["configuration"]["scheduler_agents"]["supported_runtimes"] == ["codex", "claude_code", "opencode", "pi"]
	assert contract["streaming"]["processor"] == "bytewax"
	assert contract["ui"]["routes"]
	assert contract["theme"]["name"] == "schd_scheduler_ops"


def test_schd_app_entrypoint_is_publishable():
	module = _load_module("schd_app", PACKAGE_DIR / "app.py")

	self_test = module.self_test()
	manifest = module.component_manifest()
	model = module.semantic_model()

	assert self_test["passed"] is True
	assert manifest["kind"] == "apg.generated_application"
	assert manifest["target"] == "python"
	assert model["format"] == "apg.semantic-model.v1"
	assert "schd" in model["capabilities"]
	assert "critical_job_requires_monitoring" in model["rules"]
	assert model["capabilities"]["schd"]["theme"]["name"] == "schd_scheduler_ops"


def test_schd_lifecycle_executes_with_guardrails():
	service = SchdService()
	tenant_id = "tenant-schd"

	calendar = service.create_calendar_policy(tenant_id, "weekday", "Africa/Nairobi", "ops-owner")
	worker = service.register_worker_pool(tenant_id, "etl-workers", "etl", 4)
	job = service.define_job(
		tenant_id,
		"daily-ledger-close",
		"python close_ledger.py",
		"finance-owner",
		criticality="critical",
		monitoring_attached=True,
		tags=["Finance", "Close"],
	)
	schedule = service.create_schedule(
		tenant_id,
		"daily-ledger-close",
		job["id"],
		calendar["id"],
		worker["id"],
		"interval",
		"Africa/Nairobi",
		"finance-owner",
		interval_minutes=60,
	)
	run = service.trigger_run(tenant_id, schedule["id"], "scheduler")
	completed = service.complete_run(tenant_id, run["id"], records_processed=25, logs=["close complete"])
	agent = service.register_scheduler_agent("agent-schd", tenant_id, "Codex Scheduler", "codex", "run_observer", schedule["id"], "ops-owner", True)
	summary = service.dashboard_summary(tenant_id)

	assert schedule["state"] == "active"
	assert schedule["next_run_hint"] == "every 60 minutes in Africa/Nairobi"
	assert completed["status"] == "succeeded"
	assert completed["event_stream"] == "bytewax"
	assert agent["runtime"] == "codex"
	assert service.list_schedules(tenant_id)[0]["owner"] == "finance-owner"
	assert summary["schedule_count"] == 1
	assert summary["succeeded_run_count"] == 1
	assert summary["agent_count"] == 1
	assert service.audit_events(tenant_id)


def test_schd_policy_failures_are_enforced():
	service = SchdService()
	tenant_id = "tenant-guardrails"

	with pytest.raises(PermissionError, match="tenant_context_required"):
		service.create_calendar_policy("", "missing-tenant", "UTC", "owner")

	with pytest.raises(PermissionError, match="timezone_required"):
		service.create_calendar_policy(tenant_id, "missing-zone", "", "owner")

	with pytest.raises(PermissionError, match="schedule_owner_required"):
		service.create_calendar_policy(tenant_id, "missing-owner", "UTC", "")

	with pytest.raises(PermissionError, match="critical_job_monitoring_required"):
		service.define_job(tenant_id, "critical", "python critical.py", "owner", criticality="critical")

	with pytest.raises(PermissionError, match="external_job_approval_required"):
		service.define_job(tenant_id, "external", "curl https://example.invalid", "owner", external_job=True)

	with pytest.raises(PermissionError, match="long_running_job_review_required"):
		service.define_job(tenant_id, "long", "python long.py", "owner", expected_runtime_minutes=900, monitoring_attached=True)

	calendar = service.create_calendar_policy(tenant_id, "manual", "UTC", "owner")
	worker = service.register_worker_pool(tenant_id, "manual-workers", "manual", 1)
	job = service.define_job(tenant_id, "manual-job", "python manual.py", "owner", monitoring_attached=True)
	with pytest.raises(PermissionError, match="manual_run_reason_required"):
		service.create_schedule(tenant_id, "manual", job["id"], calendar["id"], worker["id"], "manual", "UTC", "owner")

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		schedule = service.create_schedule(tenant_id, "interval", job["id"], calendar["id"], worker["id"], "interval", "UTC", "owner", interval_minutes=10)
		service.trigger_run(tenant_id, schedule["id"], "owner", event_stream="legacy_bus")

	with pytest.raises(PermissionError, match="scheduler_agent_runtime_not_supported"):
		service.register_scheduler_agent("agent-bad", tenant_id, "Bad Agent", "unknown", "run_observer", job["id"], "owner", True)

	with pytest.raises(PermissionError, match="bytewax_event_stream_required"):
		service.validate_batch_mutation("legacy_bus")


def test_schd_recovery_and_state_guardrails():
	service = SchdService()
	tenant_id = "tenant-recovery"
	calendar = service.create_calendar_policy(tenant_id, "weekday", "UTC", "owner")
	worker = service.register_worker_pool(tenant_id, "workers", "default", 2)
	job = service.define_job(tenant_id, "hourly", "python hourly.py", "owner", monitoring_attached=True, max_attempts=3)
	schedule = service.create_schedule(tenant_id, "hourly", job["id"], calendar["id"], worker["id"], "interval", "UTC", "owner", interval_minutes=30)

	with pytest.raises(PermissionError, match="schedule_pause_reason_required"):
		service.pause_schedule(tenant_id, schedule["id"], "owner", "")

	paused = service.pause_schedule(tenant_id, schedule["id"], "owner", "maintenance")
	assert paused["state"] == "paused"
	with pytest.raises(PermissionError, match="schedule_not_runnable"):
		service.trigger_run(tenant_id, schedule["id"], "owner")
	service.resume_schedule(tenant_id, schedule["id"], "owner", "maintenance complete")

	service.change_worker_state(tenant_id, worker["id"], "offline", "owner", "patching")
	with pytest.raises(PermissionError, match="worker_pool_not_ready"):
		service.trigger_run(tenant_id, schedule["id"], "owner")
	service.change_worker_state(tenant_id, worker["id"], "ready", "owner", "patching complete")

	run = service.trigger_run(tenant_id, schedule["id"], "owner")
	failed = service.complete_run(tenant_id, run["id"], error_count=1, exit_code=1, completion_evidence_ref="evidence://failed")
	retry = service.retry_run(tenant_id, failed["id"], "owner", "retry after transient failure")
	assert retry["attempt"] == 2
	assert retry["parent_run_id"] == failed["id"]

	cancelled = service.cancel_run(tenant_id, retry["id"], "owner", "operator stop")
	assert cancelled["status"] == "cancelled"

	dead = service.dead_letter_run(tenant_id, failed["id"], "owner", "manual quarantine")
	assert dead["status"] == "dead_lettered"
	assert dead["dead_letter_reason"] == "manual quarantine"


def test_schd_view_models_expose_composable_surfaces():
	from capabilities.common.schd.views import (
		analytics_model,
		audit_trail_model,
		calendar_manager_model,
		dashboard_model,
		job_library_model,
		run_monitor_model,
		schedule_console_model,
		scheduler_agent_panel_model,
		settings_model,
		worker_dashboard_model,
	)

	service = SchdService()
	tenant_id = "tenant-view"
	calendar = service.create_calendar_policy(tenant_id, "weekday", "UTC", "owner")
	worker = service.register_worker_pool(tenant_id, "workers", "default", 2)
	job = service.define_job(tenant_id, "hourly", "python hourly.py", "owner", monitoring_attached=True)
	schedule = service.create_schedule(tenant_id, "hourly", job["id"], calendar["id"], worker["id"], "interval", "UTC", "owner", interval_minutes=30)
	run = service.trigger_run(tenant_id, schedule["id"], "owner")
	service.complete_run(tenant_id, run["id"])
	service.register_scheduler_agent("agent-view", tenant_id, "Codex Scheduler", "codex", "calendar_auditor", schedule["id"], "owner", True)

	assert dashboard_model(service, tenant_id)["summary"]["schedule_count"] == 1
	assert schedule_console_model(service, tenant_id)["actions"] == ["create_schedule", "pause_schedule", "resume_schedule", "disable_schedule", "trigger_run"]
	assert job_library_model(service, tenant_id)["jobs"][0]["name"] == "hourly"
	assert run_monitor_model(service, tenant_id)["runs"][0]["status"] == "succeeded"
	assert worker_dashboard_model(service, tenant_id)["worker_pools"][0]["state"] == "ready"
	assert calendar_manager_model(service, tenant_id)["calendars"][0]["timezone"] == "UTC"
	assert scheduler_agent_panel_model(service, tenant_id)["agents"][0]["runtime"] == "codex"
	assert audit_trail_model(service, tenant_id)["streaming_topic"] == "apg.schd.lifecycle"
	assert analytics_model(service, tenant_id)["run_health"]["succeeded"] == 1
	assert settings_model(service, tenant_id)["theme"]["name"] == "schd_scheduler_ops"
