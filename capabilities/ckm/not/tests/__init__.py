#!/usr/bin/env python3
"""
Test Suite for APG Notification System

This package contains comprehensive tests for all notification system components
including unit tests, integration tests, and end-to-end tests.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import sys
import os
from pathlib import Path

# Add the notification capability to the Python path
notification_path = Path(__file__).parent.parent
if str(notification_path) not in sys.path:
    sys.path.insert(0, str(notification_path))

# Test configuration
TEST_CONFIG = {
    'test_tenant_id': 'test-tenant-12345',
    'test_user_id': 'test-user-67890',
    'redis_url': 'redis://localhost:6379/15',  # Use DB 15 for tests
    'database_url': 'sqlite:///test_notifications.db',
    'mock_external_services': True,
    'log_level': 'DEBUG'
}

# Test fixtures and utilities
from .fixtures import *
from .utils import *