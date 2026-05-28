"""Minimal-runtime regressions for composition orchestration service."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from capabilities.composition.orchestration.service import (
	NativeWorkflowService,
	WorkflowDefinition,
	WorkflowEngine,
	WorkflowInstance,
	WorkflowStatus,
	redis,
)


def test_orchestration_service_imports_without_optional_runtime_sdks():
	"""The orchestration capability should import in a dependency-light APG venv."""
	assert WorkflowEngine.NATIVE.value == "native"
	assert type(redis.from_url("redis://memory")).__name__ == "_InMemoryRedis"


@pytest.mark.asyncio
async def test_native_workflow_executes_python_tasks_without_external_engines():
	"""Native workflows should be executable without Prefect, Celery, Airflow, or Redis servers."""
	workflow = WorkflowDefinition(
		workflow_id="wf_minimal",
		name="Minimal Native Workflow",
		description=None,
		version="1.0.0",
		engine=WorkflowEngine.NATIVE,
		tasks=[
			{
				"id": "derive_total",
				"name": "Derive total",
				"type": "python",
				"code": "result = input_data['base'] + 7",
			}
		],
		dependencies={},
		triggers=[],
		variables={},
		timeout_seconds=30,
		retry_config={"fail_fast": True},
		metadata={},
	)
	instance = WorkflowInstance(
		instance_id="inst_minimal",
		workflow_id=workflow.workflow_id,
		status=WorkflowStatus.PENDING,
		current_tasks=[],
		completed_tasks=[],
		failed_tasks=[],
		context={"base": 35},
		started_at=datetime.now(timezone.utc),
		completed_at=None,
		error_message=None,
		execution_logs=[],
	)
	service = NativeWorkflowService(db_session=None, redis_client=redis.from_url("redis://memory"))

	await service.execute_workflow(workflow, instance)

	assert instance.status == WorkflowStatus.COMPLETED
	assert instance.completed_tasks == ["derive_total"]
	assert instance.failed_tasks == []
	assert instance.context["task_derive_total_result"] == 42
