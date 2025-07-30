#!/usr/bin/env python3
"""
Test Utilities for APG Notification System

Common utilities and helpers for notification system tests.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, AsyncMock, patch
import pytest
from contextlib import asynccontextmanager
import uuid

from . import TEST_CONFIG

class TestTimer:
    """Simple timer for performance testing"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class MockWebhookServer:
    """Mock webhook server for testing webhook deliveries"""
    
    def __init__(self):
        self.received_webhooks = []
        self.response_status = 200
        self.response_delay = 0.0
    
    async def receive_webhook(self, data: Dict[str, Any]) -> int:
        """Simulate receiving a webhook"""
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        
        self.received_webhooks.append({
            'data': data,
            'timestamp': datetime.utcnow(),
            'headers': {'content-type': 'application/json'}
        })
        
        return self.response_status
    
    def set_response(self, status: int = 200, delay: float = 0.0):
        """Configure webhook response"""
        self.response_status = status
        self.response_delay = delay
    
    def get_received_count(self) -> int:
        """Get number of received webhooks"""
        return len(self.received_webhooks)
    
    def get_last_webhook(self) -> Optional[Dict[str, Any]]:
        """Get the last received webhook"""
        return self.received_webhooks[-1] if self.received_webhooks else None
    
    def clear(self):
        """Clear received webhooks"""
        self.received_webhooks.clear()

class TestDataBuilder:
    """Builder for creating complex test data structures"""
    
    def __init__(self):
        self.data = {}
    
    def with_user(self, user_id: str = None, **kwargs) -> 'TestDataBuilder':
        """Add user data"""
        user_id = user_id or f"test-user-{uuid.uuid4().hex[:8]}"
        self.data['user'] = {
            'user_id': user_id,
            'email': f'{user_id}@test.com',
            'phone': '+1-555-123-4567',
            'name': f'Test User {user_id}',
            'tenant_id': TEST_CONFIG['test_tenant_id'],
            **kwargs
        }
        return self
    
    def with_template(self, template_id: str = None, **kwargs) -> 'TestDataBuilder':
        """Add template data"""
        template_id = template_id or f"template-{uuid.uuid4().hex[:8]}"
        self.data['template'] = {
            'id': template_id,
            'name': f'Test Template {template_id}',
            'subject_template': 'Test Subject - {{user_name}}',
            'text_template': 'Test message for {{user_name}}',
            'tenant_id': TEST_CONFIG['test_tenant_id'],
            **kwargs
        }
        return self
    
    def with_notification_request(self, **kwargs) -> 'TestDataBuilder':
        """Add notification request data"""
        self.data['request'] = {
            'template_id': self.data.get('template', {}).get('id', 'default-template'),
            'user_id': self.data.get('user', {}).get('user_id', 'default-user'),
            'channels': ['email'],
            'context': {'user_name': 'Test User'},
            'tenant_id': TEST_CONFIG['test_tenant_id'],
            **kwargs
        }
        return self
    
    def with_geofence(self, **kwargs) -> 'TestDataBuilder':
        """Add geofence data"""
        self.data['geofence'] = {
            'name': 'Test Geofence',
            'center_lat': 37.7749,
            'center_lon': -122.4194,
            'radius': 100.0,
            'tenant_id': TEST_CONFIG['test_tenant_id'],
            **kwargs
        }
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build the final data structure"""
        return self.data.copy()

class AsyncTestRunner:
    """Helper for running async tests with proper event loop management"""
    
    @staticmethod
    def run(coro):
        """Run async coroutine in test"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

class MockChannelProvider:
    """Mock channel provider for testing"""
    
    def __init__(self, channel_name: str, success_rate: float = 1.0):
        self.channel_name = channel_name
        self.success_rate = success_rate
        self.sent_messages = []
        self.delivery_delay = 0.0
    
    async def send(self, recipient: str, content: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Mock send method"""
        if self.delivery_delay > 0:
            await asyncio.sleep(self.delivery_delay)
        
        # Simulate success/failure based on success rate
        import random
        success = random.random() < self.success_rate
        
        message = {
            'recipient': recipient,
            'content': content,
            'channel': self.channel_name,
            'timestamp': datetime.utcnow(),
            'success': success,
            **kwargs
        }
        
        self.sent_messages.append(message)
        
        if success:
            return {
                'status': 'delivered',
                'external_id': f'mock-{uuid.uuid4().hex[:8]}',
                'delivered_at': datetime.utcnow()
            }
        else:
            return {
                'status': 'failed',
                'error': 'Mock delivery failure',
                'failed_at': datetime.utcnow()
            }
    
    def set_success_rate(self, rate: float):
        """Set success rate for deliveries"""
        self.success_rate = max(0.0, min(1.0, rate))
    
    def set_delivery_delay(self, delay: float):
        """Set delivery delay in seconds"""
        self.delivery_delay = delay
    
    def get_sent_count(self) -> int:
        """Get number of sent messages"""
        return len(self.sent_messages)
    
    def get_successful_count(self) -> int:
        """Get number of successful deliveries"""
        return sum(1 for msg in self.sent_messages if msg['success'])
    
    def clear(self):
        """Clear sent messages"""
        self.sent_messages.clear()

def create_mock_providers(channels: List[str], success_rate: float = 1.0) -> Dict[str, MockChannelProvider]:
    """Create mock providers for multiple channels"""
    providers = {}
    for channel in channels:
        providers[channel] = MockChannelProvider(channel, success_rate)
    return providers

@asynccontextmanager
async def temporary_config(updates: Dict[str, Any]):
    """Temporarily update test configuration"""
    original_config = TEST_CONFIG.copy()
    TEST_CONFIG.update(updates)
    try:
        yield TEST_CONFIG
    finally:
        TEST_CONFIG.clear()
        TEST_CONFIG.update(original_config)

class PerformanceTracker:
    """Track performance metrics during tests"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_metric(self, name: str):
        """Start tracking a metric"""
        self.start_times[name] = time.time()
    
    def end_metric(self, name: str) -> float:
        """End tracking a metric and return duration"""
        if name not in self.start_times:
            return 0.0
        
        duration = time.time() - self.start_times[name]
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        del self.start_times[name]
        return duration
    
    def get_average(self, name: str) -> float:
        """Get average duration for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def get_max(self, name: str) -> float:
        """Get maximum duration for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return max(self.metrics[name])
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get comprehensive stats for a metric"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            'count': len(values),
            'average': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'total': sum(values)
        }

def assert_notification_delivered(result: Dict[str, Any], expected_status: str = 'delivered'):
    """Assert that notification was delivered successfully"""
    assert result is not None, "Notification result is None"
    assert 'delivery_results' in result, "No delivery results in notification result"
    
    delivery_results = result['delivery_results']
    assert len(delivery_results) > 0, "No delivery results found"
    
    for delivery in delivery_results:
        assert delivery['status'] == expected_status, f"Expected {expected_status}, got {delivery['status']}"

def assert_analytics_tracked(analytics_engine: Mock, event_type: str):
    """Assert that analytics event was tracked"""
    if hasattr(analytics_engine, 'track_delivery'):
        assert analytics_engine.track_delivery.called or analytics_engine.track_engagement.called, \
            f"Analytics {event_type} was not tracked"

def assert_security_validated(security_engine: Mock):
    """Assert that security validation was performed"""
    if hasattr(security_engine, 'validate_and_secure_data'):
        assert security_engine.validate_and_secure_data.called, "Security validation was not performed"

def wait_for_async_completion(coros: List, timeout: float = 5.0):
    """Wait for multiple async operations to complete"""
    async def _wait():
        await asyncio.gather(*coros)
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(asyncio.wait_for(_wait(), timeout=timeout))
    finally:
        loop.close()

class TestMetrics:
    """Collect and analyze test metrics"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_data = {}
    
    def record_test_result(self, test_name: str, passed: bool, duration: float):
        """Record test result"""
        self.test_results[test_name] = {
            'passed': passed,
            'duration': duration,
            'timestamp': datetime.utcnow()
        }
    
    def record_performance_data(self, test_name: str, metrics: Dict[str, Any]):
        """Record performance data"""
        self.performance_data[test_name] = {
            'metrics': metrics,
            'timestamp': datetime.utcnow()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        passed_count = sum(1 for result in self.test_results.values() if result['passed'])
        total_count = len(self.test_results)
        total_duration = sum(result['duration'] for result in self.test_results.values())
        
        return {
            'tests_run': total_count,
            'tests_passed': passed_count,
            'tests_failed': total_count - passed_count,
            'pass_rate': passed_count / total_count if total_count > 0 else 0.0,
            'total_duration': total_duration,
            'average_duration': total_duration / total_count if total_count > 0 else 0.0
        }

# Global test metrics instance
test_metrics = TestMetrics()

def log_test_performance(test_name: str):
    """Decorator to log test performance"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    test_metrics.record_test_result(test_name, True, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    test_metrics.record_test_result(test_name, False, duration)
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    test_metrics.record_test_result(test_name, True, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    test_metrics.record_test_result(test_name, False, duration)
                    raise
            return sync_wrapper
    return decorator

# Assertion helpers
def assert_valid_uuid(value: str, message: str = "Invalid UUID"):
    """Assert that value is a valid UUID"""
    try:
        uuid.UUID(value)
    except ValueError:
        pytest.fail(f"{message}: {value}")

def assert_valid_email(email: str, message: str = "Invalid email"):
    """Assert that email is valid"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        pytest.fail(f"{message}: {email}")

def assert_timestamp_recent(timestamp: datetime, max_age_seconds: int = 60):
    """Assert that timestamp is recent"""
    age = (datetime.utcnow() - timestamp).total_seconds()
    assert age <= max_age_seconds, f"Timestamp too old: {age} seconds"

def assert_dict_contains(actual: Dict[str, Any], expected: Dict[str, Any]):
    """Assert that actual dict contains all expected key-value pairs"""
    for key, value in expected.items():
        assert key in actual, f"Missing key: {key}"
        assert actual[key] == value, f"Value mismatch for {key}: expected {value}, got {actual[key]}"

# Mock factory functions
def create_mock_notification_service():
    """Create mock notification service"""
    service = Mock()
    service.send_notification = AsyncMock(return_value={'id': 'mock-notification-id'})
    service.send_bulk_notifications = AsyncMock(return_value={'sent': 10, 'failed': 0})
    service.get_notification_status = AsyncMock(return_value={'status': 'delivered'})
    return service

def create_mock_analytics_service():
    """Create mock analytics service"""
    service = Mock()
    service.track_event = AsyncMock()
    service.generate_report = AsyncMock(return_value={'metrics': {}})
    return service

# Test data validation
def validate_test_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """Validate test data against schema"""
    for key, expected_type in schema.items():
        if key not in data:
            return False
        if not isinstance(data[key], expected_type):
            return False
    return True

# Database test helpers
async def cleanup_test_database():
    """Clean up test database"""
    # This would connect to test database and clean up
    pass

async def setup_test_database():
    """Set up test database with initial data"""
    # This would set up test database schema and initial data
    pass