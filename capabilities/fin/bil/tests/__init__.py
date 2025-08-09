"""
APG Billing Testing Suite

Comprehensive test suite for the billing capability including
unit tests, integration tests, and performance tests.

© 2025 Datacraft - www.datacraft.co.ke
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
from typing import Generator, Any

# Test configuration
TEST_CONFIG = {
	"database_url": "sqlite:///:memory:",
	"redis_url": "redis://localhost:6379/15",  # Use test database
	"test_timeout": 30,
	"max_test_customers": 100,
	"max_test_subscriptions": 500,
	"mock_external_services": True  # Mock payment gateways for testing
}

# Test fixtures are automatically discovered by pytest
# This file serves as the test package initializer

__all__ = ["TEST_CONFIG"]