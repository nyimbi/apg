"""
APG Workflow Orchestration Test Suite

Comprehensive test suite for workflow orchestration system with >95% code coverage.
Includes unit tests, integration tests, performance tests, and APG-specific tests.

© 2025 Datacraft. All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero <nyimbi@gmail.com>"
__copyright__ = "© 2025 Datacraft. All rights reserved."

# Test suite information
TEST_SUITE_INFO = {
	"name": "APG Workflow Orchestration Test Suite",
	"version": __version__,
	"coverage_target": 95.0,
	"test_types": [
		"unit",
		"integration", 
		"performance",
		"security",
		"api",
		"database",
		"engine",
		"service",
		"management",
		"models"
	],
	"apg_integration": True,
	"async_support": True
}

# Test markers for pytest
PYTEST_MARKERS = {
	"unit": "Unit tests for individual components",
	"integration": "Integration tests across components",
	"performance": "Performance and benchmark tests",
	"security": "Security and vulnerability tests",
	"api": "API endpoint tests",
	"database": "Database operation tests", 
	"engine": "Workflow engine tests",
	"service": "Service layer tests",
	"management": "Management layer tests",
	"models": "Data model tests",
	"apg": "APG platform integration tests",
	"slow": "Slow running tests",
	"redis": "Tests requiring Redis"
}

# Export key test utilities
from .conftest import (
	TEST_TENANT_ID, TEST_USER_ID, TestHelpers
)

__all__ = [
	"TEST_SUITE_INFO",
	"PYTEST_MARKERS", 
	"TEST_TENANT_ID",
	"TEST_USER_ID",
	"TestHelpers"
]