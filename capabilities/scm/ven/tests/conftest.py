"""Dependency-light pytest fixtures for Vendor Management tests."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
	sys.path.insert(0, str(PACKAGE_DIR))


@pytest.fixture
def tenant_id() -> str:
	return "tenant-test"


@pytest.fixture
def vendor_service():
	from service import VendorManagementLifecycleService

	return VendorManagementLifecycleService()
