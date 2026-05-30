"""Focused Cash Management test configuration.

The package-boundary tests intentionally avoid optional Redis, database, bank,
AI, and visualization dependencies. Broader integration suites should provide
their own fixtures when those providers are available.
"""

from __future__ import annotations

import pytest


TEST_TENANT_ID = "test_tenant_12345"
TEST_USER_ID = "test_user_67890"


@pytest.fixture
def tenant_id() -> str:
	return TEST_TENANT_ID


@pytest.fixture
def user_id() -> str:
	return TEST_USER_ID
