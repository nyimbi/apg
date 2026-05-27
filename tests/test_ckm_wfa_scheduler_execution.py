"""Executable scheduling regressions for CKM WFA workflow scheduler."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER_PATH = REPO_ROOT / "capabilities" / "ckm" / "wfa" / "workflow_scheduler.py"


def test_scheduler_starts_scheduled_workflows_through_runtime_boundary():
	source = SCHEDULER_PATH.read_text(encoding="utf-8")

	assert "For now, simulate execution" not in source
	assert "In production, would start actual process instance" not in source
	assert "def __init__(self, tenant_context: APGTenantContext, workflow_runtime: Any | None = None):" in source
	assert "self.workflow_runtime = workflow_runtime" in source
	assert "self.scheduled_executions: Dict[str, Dict[str, Any]] = {}" in source
	assert "execution_record = await self._start_scheduled_process(schedule)" in source
	assert "schedule.notification_settings[\"last_execution_record\"] = execution_record" in source


def test_scheduler_records_local_or_runtime_execution_artifacts():
	source = SCHEDULER_PATH.read_text(encoding="utf-8")

	assert "async def _start_scheduled_process(self, schedule: WorkflowSchedule) -> Dict[str, Any]:" in source
	assert "for method_name in (\"start_process\", \"start_process_instance\", \"create_process_instance\"):" in source
	assert "\"runtime\": \"local_scheduler\"" in source
	assert "self.scheduled_executions[execution_record[\"execution_id\"]] = execution_record" in source
	assert "def _normalize_execution_record(self, schedule: WorkflowSchedule, result: Any) -> Dict[str, Any]:" in source
	assert "record.setdefault(\"runtime\", \"workflow_runtime\")" in source
	assert "cls._instances[tenant_id].workflow_runtime = workflow_runtime" in source
