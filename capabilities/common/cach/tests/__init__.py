#!/usr/bin/env python3
"""
APG Cache Management (CACH) - Test Package
Comprehensive test suite for cache management system

Author: Nyimbi Odero
Copyright: © 2025 Datacraft
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Configure test logging
logging.getLogger('cach').setLevel(logging.DEBUG)

# Test configuration
TEST_CONFIG = {
    'cache_size_mb': 256,
    'max_entries': 10000,
    'default_ttl_seconds': 300,
    'cleanup_interval_seconds': 60,
    'ai_optimization_enabled': True,
    'predictive_caching_enabled': True,
    'security_level': 'HIGH',
    'monitoring_enabled': True
}

# Test fixtures and utilities will be imported by individual test files
__all__ = [
    'TEST_CONFIG'
]