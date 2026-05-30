"""Advanced CRM analytics tests."""

TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"

PYTEST_MARKERS = {
	"unit": "Unit tests for individual CRM components",
	"integration": "Integration tests across CRM components",
	"package": "Dependency-light package contract tests",
}

__all__ = ["PYTEST_MARKERS", "TEST_TENANT_ID", "TEST_USER_ID"]
