"""Dependency-light pytest fixtures for advanced CRM package checks."""

from __future__ import annotations

import pytest

from ..service import AdvancedCRMService


TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"


@pytest.fixture
def crm_service() -> AdvancedCRMService:
	return AdvancedCRMService()
