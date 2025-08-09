#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APG Sustainability & ESG Management - Test Suite

Comprehensive test package for ESG capability with >95% coverage,
integration testing, and APG ecosystem validation.

Copyright © 2025 Datacraft - All rights reserved.
Author: Nyimbi Odero <nyimbi@gmail.com>
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration constants
TEST_TENANT_ID = "test_tenant_esg"
TEST_USER_ID = "test_user_esg"
TEST_DATABASE_URL = "sqlite:///:memory:"

# Test data constants
SAMPLE_ESG_FRAMEWORKS = ["GRI", "SASB", "TCFD", "CSRD"]
SAMPLE_METRIC_TYPES = ["environmental", "social", "governance"]
SAMPLE_STAKEHOLDER_TYPES = ["investor", "employee", "customer", "community", "regulator"]

def pytest_configure(config):
	"""Configure pytest for ESG testing"""
	config.addinivalue_line(
		"markers", "integration: mark test as integration test"
	)
	config.addinivalue_line(
		"markers", "ai_dependent: mark test as requiring AI services"
	)
	config.addinivalue_line(
		"markers", "performance: mark test as performance test"
	)
	config.addinivalue_line(
		"markers", "security: mark test as security test"
	)
	config.addinivalue_line(
		"markers", "api: mark test as API test"
	)

class ESGTestConfig:
	"""Test configuration for ESG capability"""
	DATABASE_URL = TEST_DATABASE_URL
	TENANT_ID = TEST_TENANT_ID
	USER_ID = TEST_USER_ID
	AI_TESTING_ENABLED = True
	REAL_TIME_TESTING_ENABLED = True
	PERFORMANCE_TESTING_ENABLED = True
	
	# Test thresholds
	MIN_COVERAGE_PERCENT = 95.0
	MAX_API_RESPONSE_TIME_MS = 500
	MAX_DASHBOARD_LOAD_TIME_MS = 2000
	MAX_AI_PREDICTION_TIME_MS = 5000

# Export test utilities
__all__ = [
	"ESGTestConfig",
	"TEST_TENANT_ID",
	"TEST_USER_ID", 
	"SAMPLE_ESG_FRAMEWORKS",
	"SAMPLE_METRIC_TYPES",
	"SAMPLE_STAKEHOLDER_TYPES"
]