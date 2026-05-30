"""Workflow orchestration tests."""

TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"

PYTEST_MARKERS = {
	"unit": "Unit tests for individual components",
	"integration": "Integration tests across components",
	"package": "Dependency-light package contract tests",
}


class TestHelpers:
	"""Small helpers shared by dependency-light package tests."""

	@staticmethod
	def workflow_task(task_id: str, task_type: str = "automated", **kwargs):
		payload = {"id": task_id, "type": task_type}
		payload.update(kwargs)
		return payload


__all__ = ["PYTEST_MARKERS", "TEST_TENANT_ID", "TEST_USER_ID", "TestHelpers"]
