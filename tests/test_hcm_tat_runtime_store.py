import ast
import asyncio
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from fastapi import FastAPI
from fastapi.testclient import TestClient

from capabilities.hcm.tat.time_attendance.api import create_app, get_current_user, get_service
from capabilities.hcm.tat.time_attendance.blueprint import time_attendance_bp
from capabilities.hcm.tat.time_attendance.mobile_api import (
	_mobile_runtime_state,
	_process_photo_verification,
	_process_work_summary,
	_register_push_token,
	_send_mobile_notification,
	get_mobile_service,
	get_mobile_user,
	mobile_router,
)
from capabilities.hcm.tat.time_attendance.monitoring import (
	Alert,
	AlertManager,
	AlertSeverity,
	BusinessMetricsMonitor,
)
from capabilities.hcm.tat.time_attendance.models import (
	AIAgentType,
	LeaveType,
	ProductivityMetric,
	TimeEntryStatus,
	WorkMode,
)
from capabilities.hcm.tat.time_attendance.reporting import (
	ReportConfig,
	ReportFormat,
	ReportGenerator,
	ReportPeriod,
	ReportType,
)
from capabilities.hcm.tat.time_attendance.service import TimeAttendanceService
from capabilities.hcm.tat.time_attendance.websocket import WebSocketManager


def _run(coro):
	return asyncio.run(coro)


async def _seed_runtime(service: TimeAttendanceService, tenant_id: str = "tenant_tat") -> dict:
	entry = await service.clock_in("emp_001", tenant_id, {"device_id": "web"})
	entry.clock_in = entry.clock_in - timedelta(hours=9)
	entry.entry_date = date.today()  # pin to today even if clock_in adjustment crosses midnight
	await service._save_time_entry(entry)
	clocked_out = await service.clock_out("emp_001", tenant_id, {"device_id": "web"})

	await service.start_remote_work_session(
		"emp_001",
		tenant_id,
		{
			"location": "Home",
			"equipment": {"computer": "laptop"},
			"timezone": "UTC",
			"collaboration_platforms": ["slack"],
		},
		WorkMode.REMOTE_ONLY,
	)
	productivity = await service.track_remote_productivity(
		"emp_001",
		tenant_id,
		{"tasks_completed": 4, "active_minutes": 420},
		ProductivityMetric.TASK_COMPLETION,
	)

	agent = await service.register_ai_agent(
		"Codex",
		AIAgentType.AUTOMATION_BOT,
		["code_generation"],
		tenant_id,
		{"api_endpoints": [], "resource_limits": {}, "cost_per_hour": "0.20"},
		"admin",
	)
	await service.track_ai_agent_work(
		agent.id,
		tenant_id,
		{"completed": True, "duration_seconds": 30, "accuracy_score": 0.9},
		{"cpu_hours": 1, "memory_gb_hours": 2, "api_calls": 10},
	)

	schedule = await service.create_intelligent_schedule(
		"Ops",
		tenant_id,
		[{"days_of_week": [0, 1, 2, 3, 4], "start_time": "09:00", "end_time": "17:00"}],
		["emp_001"],
	)
	leave = await service.process_leave_request(
		"emp_001",
		tenant_id,
		LeaveType.VACATION,
		date.today(),
		date.today() + timedelta(days=1),
	)
	return {
		"time_entry_id": clocked_out.id,
		"agent_id": agent.id,
		"schedule_id": schedule.id,
		"leave_id": leave.id,
		"productivity": productivity,
	}


def test_time_attendance_service_runtime_store_supports_core_records():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()

	result = _run(_seed_runtime(service))
	dashboard = _run(service.get_analytics_dashboard("tenant_tat"))

	assert result["productivity"]["burnout_risk"] == "LOW"
	assert len(_run(service.list_time_entries("tenant_tat"))) == 1
	assert len(_run(service.list_remote_workers("tenant_tat"))) == 1
	assert len(_run(service.list_ai_agents("tenant_tat"))) == 1
	assert len(_run(service.list_schedules("tenant_tat"))) == 1
	assert len(_run(service.list_leave_requests("tenant_tat"))) == 1
	assert dashboard["workforce_distribution"]["remote_workers"] == 1
	assert dashboard["workforce_distribution"]["ai_agents"] == 1

	bulk_update = _run(
		service.bulk_update_time_entries(
			"tenant_tat",
			[result["time_entry_id"]],
			{"notes": "corrected"},
			"admin",
		)
	)
	bulk_approval = _run(
		service.bulk_approve_entries(
			"tenant_tat",
			[result["leave_id"]],
			"leave_request",
			"admin",
			action="approve",
			approval_notes="ok",
		)
	)
	assert bulk_update["updated_ids"] == [result["time_entry_id"]]
	assert bulk_approval["processed_ids"] == [result["leave_id"]]


def test_time_attendance_api_lists_are_service_backed():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service))

	app = create_app()
	app.dependency_overrides[get_current_user] = lambda: {
		"user_id": "admin",
		"tenant_id": "tenant_tat",
		"roles": ["admin"],
	}
	app.dependency_overrides[get_service] = lambda: service
	client = TestClient(app)

	time_entries = client.get("/api/human_capital_management/time_attendance/time-entries")
	remote_workers = client.get("/api/human_capital_management/time_attendance/remote-workers")
	ai_agents = client.get("/api/human_capital_management/time_attendance/ai-agents")
	dashboard = client.get("/api/human_capital_management/time_attendance/analytics/dashboard")

	assert time_entries.status_code == 200
	assert time_entries.json()["pagination"]["total"] == 1
	assert remote_workers.status_code == 200
	assert remote_workers.json()["summary"]["total_remote_workers"] == 1
	assert ai_agents.status_code == 200
	assert ai_agents.json()["summary"]["total_ai_agents"] == 1
	assert dashboard.status_code == 200
	assert dashboard.json()["data"]["workforce_distribution"]["ai_agents"] == 1


def test_mobile_api_status_and_analytics_are_service_backed():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service))

	app = FastAPI()
	app.include_router(mobile_router)
	app.dependency_overrides[get_mobile_user] = lambda: {
		"user_id": "admin",
		"tenant_id": "tenant_tat",
		"employee_id": "emp_001",
		"device_id": "device_001",
		"roles": ["employee"],
	}
	app.dependency_overrides[get_mobile_service] = lambda: service
	client = TestClient(app)

	status = client.get("/api/mobile/human_capital_management/time_attendance/quick-status")
	analytics = client.get("/api/mobile/human_capital_management/time_attendance/analytics/personal")

	assert status.status_code == 200
	assert status.json()["today_total_hours"] > 0
	assert status.json()["week_total_hours"] >= status.json()["today_total_hours"]
	assert analytics.status_code == 200
	assert analytics.json()["total_hours"] == status.json()["week_total_hours"]
	assert len(analytics.json()["daily_breakdown"]) == 7


def test_mobile_runtime_helpers_record_side_effects():
	_mobile_runtime_state["notifications"].clear()
	_mobile_runtime_state["photo_verifications"].clear()
	_mobile_runtime_state["work_summaries"].clear()
	_mobile_runtime_state["push_tokens"].clear()

	_run(_send_mobile_notification("device_001", "Clock", "ok", {"type": "clock"}))
	_run(_process_photo_verification("encoded-photo", "emp_001", "entry_001"))
	_run(_process_work_summary("Closed tickets", "entry_001", 5))
	_run(_register_push_token("user_001", "device_001", "token", "ios", {"clock": True}))

	assert _mobile_runtime_state["notifications"]["device_001"][0]["title"] == "Clock"
	assert _mobile_runtime_state["photo_verifications"]["entry_001"]["verified"] is True
	assert _mobile_runtime_state["work_summaries"]["entry_001"]["productivity_rating"] == 5
	assert _mobile_runtime_state["push_tokens"]["device_001"]["platform"] == "ios"


def test_flask_blueprint_lists_use_service_runtime_store():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service, tenant_id="tenant_blueprint"))

	app = Flask(__name__)
	app.secret_key = "test"
	app.sm = SimpleNamespace(user=SimpleNamespace(id="admin", tenant_id="tenant_blueprint"))
	app.register_blueprint(time_attendance_bp)
	client = app.test_client()

	time_entries = client.get("/api/human_capital_management/time_attendance/time-entries")
	remote_workers = client.get("/api/human_capital_management/time_attendance/remote-workers")
	ai_agents = client.get("/api/human_capital_management/time_attendance/ai-agents")

	assert time_entries.status_code == 200
	assert time_entries.json["data"]["total"] == 1
	assert time_entries.json["data"]["time_entries"][0]["employee_id"] == "emp_001"
	assert remote_workers.status_code == 200
	assert remote_workers.json["data"]["total"] == 1
	assert remote_workers.json["data"]["remote_workers"][0]["employee_id"] == "emp_001"
	assert ai_agents.status_code == 200
	assert ai_agents.json["data"]["total"] == 1
	assert ai_agents.json["data"]["ai_agents"][0]["agent_name"] == "Codex"


def test_reporting_generator_uses_service_runtime_store():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service, tenant_id="tenant_reporting"))
	generator = ReportGenerator(service)
	start = date.today() - timedelta(days=7)
	end = date.today() + timedelta(days=1)

	timesheet = _run(
		generator.generate_report(
			ReportConfig(
				report_type=ReportType.TIMESHEET,
				format=ReportFormat.JSON,
				period=ReportPeriod.CUSTOM,
				start_date=start,
				end_date=end,
				tenant_id="tenant_reporting",
			),
			"admin",
		)
	)
	ai_agents = _run(
		generator.generate_report(
			ReportConfig(
				report_type=ReportType.AI_AGENT_UTILIZATION,
				format=ReportFormat.JSON,
				period=ReportPeriod.CUSTOM,
				start_date=start,
				end_date=end,
				tenant_id="tenant_reporting",
			),
			"admin",
		)
	)

	assert timesheet["success"] is True
	assert timesheet["data"]["report_data"]["summary"]["total_records"] == 1
	assert timesheet["data"]["report_data"]["records"][0]["employee_id"] == "emp_001"
	assert ai_agents["success"] is True
	assert ai_agents["data"]["report_data"]["summary"]["total_agents"] == 1
	assert ai_agents["data"]["report_data"]["summary"]["total_tasks_completed"] == 1


def test_business_monitoring_uses_service_runtime_store():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service, tenant_id="tenant_monitoring"))
	monitor = BusinessMetricsMonitor(service)

	metrics = _run(monitor.collect_business_metrics("tenant_monitoring"))
	health = _run(monitor.generate_health_report("tenant_monitoring"))

	assert metrics["active_employees"] == 1
	assert metrics["clock_in_rate_today"] == 1.0
	assert metrics["remote_workers_active"] == 1
	assert metrics["ai_agents_active"] == 1
	assert metrics["approval_pending_count"] >= 1
	assert health["tenant_id"] == "tenant_monitoring"
	assert health["business_metrics"]["active_employees"] == 1


def test_alert_manager_records_configured_notification_channels():
	manager = AlertManager()
	channel = manager.configure_notification_channel("email", "ops@example.com", priority="high")
	alert = Alert(
		id="alert_001",
		title="High Overtime",
		description="Overtime exceeded threshold",
		severity=AlertSeverity.WARNING,
		metric_name="overtime_hours",
		current_value=12.0,
		threshold_value=8.0,
		timestamp=datetime.utcnow(),
		tenant_id="tenant_monitoring",
	)

	_run(manager.send_alert(alert))

	assert channel["type"] == "email"
	assert list(manager.notification_history)[0]["channel"] == "websocket"
	assert list(manager.notification_history)[1]["channel"] == "email"
	assert list(manager.notification_history)[1]["status"] == "queued"


def test_websocket_dashboard_data_uses_service_runtime_store():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	_run(_seed_runtime(service, tenant_id="tenant_websocket"))
	manager = WebSocketManager()
	manager.set_service(service)

	overview = _run(manager._generate_dashboard_data("overview", "tenant_websocket", "admin"))
	remote_work = _run(manager._generate_dashboard_data("remote_work", "tenant_websocket", "admin"))
	ai_agents = _run(manager._generate_dashboard_data("ai_agents", "tenant_websocket", "admin"))

	assert overview["active_employees"] == 1
	assert overview["remote_workers"] == 1
	assert overview["ai_agents_active"] == 1
	assert overview["recent_activities"][0]["employee_id"] == "emp_001"
	assert remote_work["total_remote_workers"] == 1
	assert remote_work["active_sessions"] == 1
	assert remote_work["top_performers"][0]["employee_id"] == "emp_001"
	assert ai_agents["total_agents"] == 1
	assert ai_agents["active_agents"] == 1
	assert ai_agents["tasks_completed_today"] == 1


def test_compliance_rules_detect_runtime_time_entry_violations():
	TimeAttendanceService.reset_runtime_store()
	service = TimeAttendanceService()
	seed = _run(_seed_runtime(service, tenant_id="tenant_compliance"))
	entry = _run(service.list_time_entries("tenant_compliance"))[0]
	entry.total_hours = Decimal("17")
	entry.regular_hours = Decimal("8")
	entry.overtime_hours = Decimal("9")
	entry.break_minutes = 0
	entry.status = TimeEntryStatus.SUBMITTED
	entry.approved_by = None
	_run(service._save_time_entry(entry))

	enforcement = _run(service.enforce_compliance_rules("tenant_compliance"))
	analytics = _run(service.generate_workforce_predictions("tenant_compliance", 7, None))

	assert enforcement["violations_detected"] >= 3
	assert enforcement["corrections_applied"] >= 2
	assert enforcement["compliance_score"] < 1.0
	assert {violation["rule_code"] for violation in enforcement["violations"]} >= {
		"DAILY_MAX_HOURS",
		"MINIMUM_BREAK",
		"OVERTIME_APPROVAL",
	}
	assert analytics.compliance_risks
	assert seed["time_entry_id"] == entry.id


def test_time_attendance_service_has_no_missing_private_helper_calls():
	source_path = Path("capabilities/hcm/tat/time_attendance/service.py")
	tree = ast.parse(source_path.read_text())
	service_class = next(
		node for node in tree.body
		if isinstance(node, ast.ClassDef) and node.name == "TimeAttendanceService"
	)
	defined = {
		node.name for node in service_class.body
		if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
	}
	called = set()
	for node in ast.walk(service_class):
		if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
			continue
		if isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
			if node.func.attr.startswith("_"):
				called.add(node.func.attr)

	assert sorted(called - defined) == []
