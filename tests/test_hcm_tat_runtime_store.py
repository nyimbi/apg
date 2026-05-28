import ast
import asyncio
from datetime import date, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from capabilities.hcm.tat.time_attendance.api import create_app, get_current_user, get_service
from capabilities.hcm.tat.time_attendance.models import (
	AIAgentType,
	LeaveType,
	ProductivityMetric,
	WorkMode,
)
from capabilities.hcm.tat.time_attendance.service import TimeAttendanceService


def _run(coro):
	return asyncio.run(coro)


async def _seed_runtime(service: TimeAttendanceService, tenant_id: str = "tenant_tat") -> dict:
	entry = await service.clock_in("emp_001", tenant_id, {"device_id": "web"})
	entry.clock_in = entry.clock_in - timedelta(hours=9)
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
