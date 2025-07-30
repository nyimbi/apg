#!/usr/bin/env python3
"""
Test Fixtures for APG Notification System

Common fixtures and test data for notification system tests.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, MagicMock
import tempfile
import os

# Import notification system components
from ..models import *
from ..service import NotificationService, create_notification_service
from ..analytics_engine import AnalyticsEngine
from ..security_engine import SecurityEngine
from ..geofencing_engine import GeofencingEngine, Location
from ..personalization.core import DeepPersonalizationEngine
from . import TEST_CONFIG

@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    redis_mock = Mock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=False)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=True)
    redis_mock.expire = AsyncMock(return_value=True)
    return redis_mock

@pytest.fixture
def mock_database():
    """Mock database connection"""
    db_mock = Mock()
    db_mock.execute = AsyncMock()
    db_mock.fetch = AsyncMock(return_value=[])
    db_mock.fetchrow = AsyncMock(return_value=None)
    db_mock.transaction = AsyncMock()
    return db_mock

@pytest.fixture
def test_config():
    """Test configuration"""
    return TEST_CONFIG.copy()

@pytest.fixture
def sample_notification_template():
    """Sample notification template for testing"""
    return UltimateNotificationTemplate(
        id="test-template-001",
        name="Test Welcome Template",
        subject_template="Welcome {{user_name}}!",
        text_template="Hi {{user_name}}, welcome to {{company_name}}!",
        html_template="<h1>Welcome {{user_name}}!</h1><p>Welcome to {{company_name}}!</p>",
        tenant_id=TEST_CONFIG['test_tenant_id']
    )

@pytest.fixture
def sample_user_profile():
    """Sample user profile for testing"""
    return UserProfile(
        user_id=TEST_CONFIG['test_user_id'],
        email="test@example.com",
        phone="+1-555-123-4567",
        name="Test User",
        preferences={
            'email_enabled': True,
            'sms_enabled': True,
            'push_enabled': True,
            'frequency': 'normal',
            'quiet_hours': {'start': 22, 'end': 7}
        },
        tenant_id=TEST_CONFIG['test_tenant_id']
    )

@pytest.fixture
def sample_notification_request():
    """Sample notification request for testing"""
    return NotificationRequest(
        template_id="test-template-001",
        user_id=TEST_CONFIG['test_user_id'],
        channels=[DeliveryChannel.EMAIL, DeliveryChannel.SMS],
        context={
            'user_name': 'Test User',
            'company_name': 'Test Company'
        },
        priority=NotificationPriority.NORMAL,
        tenant_id=TEST_CONFIG['test_tenant_id']
    )

@pytest.fixture
def sample_bulk_request():
    """Sample bulk notification request for testing"""
    return BulkNotificationRequest(
        template_id="test-template-001",
        user_ids=[f"user-{i}" for i in range(1, 11)],
        channels=[DeliveryChannel.EMAIL],
        context={'company_name': 'Test Company'},
        tenant_id=TEST_CONFIG['test_tenant_id']
    )

@pytest.fixture
def sample_campaign():
    """Sample campaign for testing"""
    return Campaign(
        id="test-campaign-001",
        name="Test Welcome Campaign",
        type=CampaignType.TRANSACTIONAL,
        status=CampaignStatus.ACTIVE,
        schedule_type=ScheduleType.IMMEDIATE,
        tenant_id=TEST_CONFIG['test_tenant_id'],
        templates=["test-template-001"],
        target_criteria={'segment': 'new_users'}
    )

@pytest.fixture
def sample_location():
    """Sample location for geofencing tests"""
    return Location(
        latitude=37.7749,
        longitude=-122.4194,
        accuracy=10.0,
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def sample_locations():
    """Sample location sequence for testing movement"""
    base_lat, base_lon = 37.7749, -122.4194
    locations = []
    
    for i in range(10):
        lat_offset = (i * 0.001)  # Move north
        lon_offset = (i * 0.001)  # Move east
        
        locations.append(Location(
            latitude=base_lat + lat_offset,
            longitude=base_lon + lon_offset,
            accuracy=10.0,
            timestamp=datetime.utcnow() + timedelta(minutes=i)
        ))
    
    return locations

@pytest.fixture
def mock_channel_providers():
    """Mock channel providers for testing"""
    providers = {}
    
    for channel in DeliveryChannel:
        provider = Mock()
        provider.send = AsyncMock(return_value=DeliveryResult(
            channel=channel,
            status=DeliveryStatus.DELIVERED,
            external_id=f"ext-{channel.value}-123",
            delivered_at=datetime.utcnow()
        ))
        provider.validate_recipient = Mock(return_value=True)
        provider.get_capabilities = Mock(return_value={'supports_rich_media': True})
        providers[channel] = provider
    
    return providers

@pytest.fixture
def mock_personalization_engine():
    """Mock personalization engine"""
    engine = Mock(spec=DeepPersonalizationEngine)
    engine.personalize_message = AsyncMock(return_value={
        'personalized_content': {
            'subject': 'Personalized Subject',
            'text': 'Personalized message content',
            'html': '<p>Personalized HTML content</p>'
        },
        'quality_score': 0.85,
        'strategies_applied': ['neural_content', 'behavioral_adaptive'],
        'processing_time_ms': 45
    })
    return engine

@pytest.fixture
def mock_analytics_engine():
    """Mock analytics engine"""
    engine = Mock(spec=AnalyticsEngine)
    engine.track_delivery = AsyncMock()
    engine.track_engagement = AsyncMock()
    engine.generate_report = AsyncMock(return_value={
        'summary': {'total_sent': 100, 'delivered': 95, 'opened': 60, 'clicked': 25},
        'metrics': {'delivery_rate': 0.95, 'open_rate': 0.63, 'click_rate': 0.42}
    })
    return engine

@pytest.fixture
def mock_security_engine():
    """Mock security engine"""
    engine = Mock(spec=SecurityEngine)
    engine.validate_and_secure_data = AsyncMock(return_value={
        'data': {'secure': True},
        'compliance': {'compliant': True, 'violations': []},
        'security_applied': True
    })
    engine.authenticate_session = AsyncMock(return_value={
        'authenticated': True,
        'user_id': TEST_CONFIG['test_user_id']
    })
    return engine

@pytest.fixture
async def notification_service(
    mock_redis,
    mock_database,
    mock_channel_providers,
    mock_personalization_engine,
    mock_analytics_engine,
    mock_security_engine,
    test_config
):
    """Create notification service with mocked dependencies"""
    
    # Create service with mocked components
    service = create_notification_service(
        tenant_id=test_config['test_tenant_id'],
        config={
            'redis_client': mock_redis,
            'database': mock_database,
            'channel_providers': mock_channel_providers,
            'personalization_engine': mock_personalization_engine,
            'analytics_engine': mock_analytics_engine,
            'security_engine': mock_security_engine
        }
    )
    
    return service

@pytest.fixture
def sample_analytics_data():
    """Sample analytics data for testing"""
    return {
        'delivery_metrics': {
            'total_sent': 1000,
            'delivered': 950,
            'failed': 30,
            'bounced': 20,
            'delivery_rate': 0.95
        },
        'engagement_metrics': {
            'opened': 600,
            'clicked': 240,
            'converted': 48,
            'open_rate': 0.63,
            'click_rate': 0.40,
            'conversion_rate': 0.20
        },
        'channel_performance': {
            'email': {'sent': 600, 'delivered': 580, 'opened': 350},
            'sms': {'sent': 200, 'delivered': 195, 'opened': 120},
            'push': {'sent': 200, 'delivered': 175, 'opened': 130}
        },
        'time_series': [
            {'timestamp': '2025-01-01T00:00:00Z', 'sent': 100, 'delivered': 95},
            {'timestamp': '2025-01-01T01:00:00Z', 'sent': 120, 'delivered': 115},
            {'timestamp': '2025-01-01T02:00:00Z', 'sent': 110, 'delivered': 108}
        ]
    }

@pytest.fixture
def sample_security_data():
    """Sample security test data"""
    return {
        'valid_data': {
            'email': 'test@example.com',
            'message': 'Clean test message',
            'phone': '+1-555-123-4567'
        },
        'malicious_data': {
            'email': 'test+<script>alert("xss")</script>@example.com',
            'message': 'Message with <script>alert("malicious")</script> content',
            'phone': '+1-555-123-4567; DROP TABLE users;'
        },
        'invalid_data': {
            'email': 'not-an-email',
            'message': '',
            'phone': 'not-a-phone'
        }
    }

@pytest.fixture
def sample_geofence_data():
    """Sample geofencing test data"""
    return {
        'circular_geofence': {
            'name': 'Test Office',
            'center': Location(37.7749, -122.4194),
            'radius': 100.0,
            'notification_config': {
                'channels': ['push'],
                'enter_template': {
                    'subject': 'Welcome to office!',
                    'message': 'You have arrived at work'
                }
            }
        },
        'polygon_geofence': {
            'name': 'Test Area',
            'vertices': [
                Location(37.7749, -122.4194),
                Location(37.7759, -122.4194),
                Location(37.7759, -122.4184),
                Location(37.7749, -122.4184)
            ]
        }
    }

@pytest.fixture
def temp_file():
    """Create temporary file for testing"""
    fd, path = tempfile.mkstemp()
    yield path
    os.close(fd)
    os.unlink(path)

@pytest.fixture
def temp_directory():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests"""
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@pytest.fixture
def mock_external_api():
    """Mock external API responses"""
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)
        
        async def json(self):
            return self.json_data
        
        async def text(self):
            return self.text
    
    return MockResponse

@pytest.fixture
def performance_test_data():
    """Data for performance testing"""
    return {
        'small_load': {
            'users': 100,
            'notifications_per_user': 1,
            'expected_duration_seconds': 5
        },
        'medium_load': {
            'users': 1000,
            'notifications_per_user': 5,
            'expected_duration_seconds': 30
        },
        'large_load': {
            'users': 10000,
            'notifications_per_user': 10,
            'expected_duration_seconds': 300
        }
    }

# Async fixture helpers
def async_fixture(func):
    """Decorator to create async fixtures"""
    return pytest.fixture(func)

# Test data generators
def generate_test_users(count: int = 10) -> List[Dict[str, Any]]:
    """Generate test user data"""
    users = []
    for i in range(count):
        users.append({
            'user_id': f'test-user-{i:04d}',
            'email': f'user{i}@test.com',
            'phone': f'+1-555-{i:04d}',
            'name': f'Test User {i}',
            'tenant_id': TEST_CONFIG['test_tenant_id']
        })
    return users

def generate_test_templates(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test template data"""
    templates = []
    for i in range(count):
        templates.append({
            'id': f'test-template-{i:03d}',
            'name': f'Test Template {i}',
            'subject_template': f'Test Subject {i} - {{user_name}}',
            'text_template': f'Test message {i} for {{user_name}}',
            'tenant_id': TEST_CONFIG['test_tenant_id']
        })
    return templates

# Cleanup helpers
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test"""
    yield
    # Cleanup code here if needed
    pass