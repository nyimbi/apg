"""Dependency-light pytest fixtures for workflow orchestration package checks."""

from __future__ import annotations

import pytest

from ..service import WorkflowOrchestrationService


TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"


@pytest.fixture
def workflow_service() -> WorkflowOrchestrationService:
	return WorkflowOrchestrationService()


class TestHelpers:
	"""Small helper factory for orchestration tests."""

	@staticmethod
	def automated_task(task_id: str, depends_on: list[str] | None = None) -> dict[str, object]:
		return {"id": task_id, "type": "automated", "handler": f"{task_id}.handler", "depends_on": depends_on or []}

	@staticmethod
	def human_task(task_id: str, assignee: str, depends_on: list[str] | None = None) -> dict[str, object]:
		return {"id": task_id, "type": "human", "assignee": assignee, "depends_on": depends_on or []}
