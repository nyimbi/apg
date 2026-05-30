"""Dependency-light pytest fixtures for capability registry package checks."""

from __future__ import annotations

import pytest

from .service import CompositionRegistryService


TEST_TENANT_ID = "test_tenant"
TEST_USER_ID = "test_user"


@pytest.fixture
def registry_service() -> CompositionRegistryService:
	return CompositionRegistryService()
