#!/usr/bin/env python3
"""
Pytest Configuration for APG Notification System Tests

Global configuration, fixtures, and test setup for the notification system test suite.

Author: Nyimbi Odero
Company: Datacraft
Copyright: © 2025
"""

import pytest
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock
import tempfile
import shutil

# Add the notification capability to the Python path
notification_path = Path(__file__).parent.parent
if str(notification_path) not in sys.path:
    sys.path.insert(0, str(notification_path))

# Import test configuration
from . import TEST_CONFIG

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_logs.log')
    ]
)

# Suppress verbose logging from external libraries during tests
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests"""
    
    # Create test directories
    test_dirs = [
        'test_data',
        'test_logs',
        'test_cache',
        'test_uploads'
    ]
    
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['CACHE_TTL'] = '60'  # Short cache TTL for tests
    
    yield
    
    # Cleanup after all tests
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name, ignore_errors=True)

# Event loop configuration for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

# Async test marker configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "external: marks tests that require external services"
    )

# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    
    for item in items:
        # Add slow marker to tests with "slow" in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.name.lower() or "perf" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add external marker to tests requiring external services
        if any(keyword in item.name.lower() for keyword in ["webhook", "api", "external"]):
            item.add_marker(pytest.mark.external)

# Session-scoped fixtures
@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL"""
    return TEST_CONFIG['database_url']

@pytest.fixture(scope="session") 
def test_redis_url():
    """Test Redis URL"""
    return TEST_CONFIG['redis_url']

@pytest.fixture(scope="session")
def test_tenant_id():
    """Test tenant ID"""
    return TEST_CONFIG['test_tenant_id']

@pytest.fixture(scope="session")
def test_user_id():
    """Test user ID"""
    return TEST_CONFIG['test_user_id']

# Function-scoped fixtures for test isolation
@pytest.fixture
def isolated_test_config():
    """Isolated test configuration for each test"""
    return TEST_CONFIG.copy()

@pytest.fixture
def temp_directory():
    """Create temporary directory for test"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def temp_file():
    """Create temporary file for test"""
    fd, path = tempfile.mkstemp()
    try:
        yield path
    finally:
        os.close(fd)
        os.unlink(path)

# Mock fixtures
@pytest.fixture
def mock_async_client():
    """Mock async HTTP client"""
    client = Mock()
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()
    client.delete = AsyncMock()
    client.close = AsyncMock()
    return client

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection"""
    ws = Mock()
    ws.send = AsyncMock()
    ws.receive = AsyncMock()
    ws.close = AsyncMock()
    ws.closed = False
    return ws

@pytest.fixture
def mock_email_client():
    """Mock email client"""
    client = Mock()
    client.send_email = AsyncMock(return_value={'message_id': 'mock-email-123'})
    client.verify_email = AsyncMock(return_value=True)
    return client

@pytest.fixture
def mock_sms_client():
    """Mock SMS client"""
    client = Mock()
    client.send_sms = AsyncMock(return_value={'message_id': 'mock-sms-123'})
    client.verify_phone = AsyncMock(return_value=True)
    return client

# Performance testing fixtures
@pytest.fixture
def performance_metrics():
    """Performance metrics collector"""
    from .utils import PerformanceTracker
    return PerformanceTracker()

@pytest.fixture
def benchmark_timer():
    """Benchmark timer for performance tests"""
    from .utils import TestTimer
    return TestTimer()

# Database fixtures
@pytest.fixture
async def test_database():
    """Test database connection"""
    # In a real implementation, this would create a test database connection
    # For now, return a mock
    db = Mock()
    db.execute = AsyncMock()
    db.fetch = AsyncMock(return_value=[])
    db.fetchrow = AsyncMock(return_value=None)
    db.transaction = AsyncMock()
    return db

@pytest.fixture
async def clean_database(test_database):
    """Clean database before and after test"""
    # Clean before test
    await test_database.execute("DELETE FROM test_table")
    
    yield test_database
    
    # Clean after test
    await test_database.execute("DELETE FROM test_table")

# Cache fixtures
@pytest.fixture
def test_cache():
    """Test cache client"""
    cache = Mock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(return_value=True)
    cache.delete = AsyncMock(return_value=True)
    cache.clear = AsyncMock(return_value=True)
    return cache

# File system fixtures
@pytest.fixture
def test_file_storage():
    """Test file storage"""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = Mock()
        storage.base_path = temp_dir
        storage.save_file = Mock()
        storage.load_file = Mock()
        storage.delete_file = Mock()
        yield storage

# Network fixtures
@pytest.fixture
def mock_network_delay():
    """Mock network delay for testing network conditions"""
    import time
    
    async def delay(seconds=0.1):
        await asyncio.sleep(seconds)
    
    return delay

# Test data generation fixtures
@pytest.fixture
def test_data_generator():
    """Test data generator utilities"""
    from .utils import TestDataBuilder
    return TestDataBuilder()

@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing"""
    def generate(size=1000):
        return [
            {
                'id': f'item_{i:06d}',
                'name': f'Test Item {i}',
                'value': i * 10,
                'active': i % 2 == 0
            }
            for i in range(size)
        ]
    return generate

# Error simulation fixtures
@pytest.fixture
def error_simulator():
    """Simulate various error conditions"""
    class ErrorSimulator:
        def network_error(self):
            return ConnectionError("Network connection failed")
        
        def timeout_error(self):
            return asyncio.TimeoutError("Operation timed out")
        
        def permission_error(self):
            return PermissionError("Permission denied")
        
        def validation_error(self):
            return ValueError("Invalid data provided")
        
        def server_error(self):
            return Exception("Internal server error")
    
    return ErrorSimulator()

# Concurrent testing fixtures
@pytest.fixture
def concurrency_tester():
    """Utilities for testing concurrent operations"""
    
    class ConcurrencyTester:
        async def run_concurrent(self, tasks, max_concurrent=10):
            """Run tasks with limited concurrency"""
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            return await asyncio.gather(*[limited_task(task) for task in tasks])
        
        async def stress_test(self, operation, count=100, duration=60):
            """Run stress test for specified duration"""
            start_time = asyncio.get_event_loop().time()
            tasks = []
            
            while asyncio.get_event_loop().time() - start_time < duration and len(tasks) < count:
                tasks.append(asyncio.create_task(operation()))
                await asyncio.sleep(0.01)  # Small delay between task creation
            
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    return ConcurrencyTester()

# Monitoring fixtures
@pytest.fixture
def test_monitor():
    """Test monitoring and observability"""
    
    class TestMonitor:
        def __init__(self):
            self.metrics = {}
            self.logs = []
            self.events = []
        
        def record_metric(self, name: str, value: float, tags: Dict[str, str] = None):
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append({'value': value, 'tags': tags or {}, 'timestamp': asyncio.get_event_loop().time()})
        
        def log_event(self, event: str, data: Dict[str, Any] = None):
            self.events.append({'event': event, 'data': data or {}, 'timestamp': asyncio.get_event_loop().time()})
        
        def get_metric_summary(self, name: str) -> Dict[str, float]:
            if name not in self.metrics:
                return {}
            
            values = [m['value'] for m in self.metrics[name]]
            return {
                'count': len(values),
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
    
    return TestMonitor()

# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield
    
    # Cleanup any remaining async tasks
    try:
        pending_tasks = [task for task in asyncio.all_tasks() if not task.done()]
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            # Wait briefly for cancellation
            asyncio.get_event_loop().run_until_complete(asyncio.sleep(0.1))
    except RuntimeError:
        # Event loop might be closed
        pass

# Test reporting fixtures
@pytest.fixture(scope="session", autouse=True)
def test_session_summary(request):
    """Generate test session summary"""
    yield
    
    # Generate summary after all tests complete
    if hasattr(request.config, 'option') and getattr(request.config.option, 'verbose', 0) > 0:
        print("\n" + "="*80)
        print("APG NOTIFICATION SYSTEM TEST SUMMARY")
        print("="*80)
        print("Test session completed successfully")
        print("For detailed results, check test_logs.log")
        print("="*80)

# Parameterized test data
@pytest.fixture(params=[
    'email',
    'sms', 
    'push',
    'webhook'
])
def channel_type(request):
    """Parameterized channel types for testing"""
    return request.param

@pytest.fixture(params=[
    'low',
    'normal',
    'high',
    'urgent',
    'critical'
])
def priority_level(request):
    """Parameterized priority levels for testing"""
    return request.param

@pytest.fixture(params=[10, 100, 1000])
def test_scale(request):
    """Parameterized test scales for performance testing"""
    return request.param

# Skip conditions
def pytest_runtest_setup(item):
    """Setup conditions for running tests"""
    
    # Skip external tests if external services not available
    if item.get_closest_marker("external"):
        if os.environ.get("SKIP_EXTERNAL_TESTS", "false").lower() == "true":
            pytest.skip("External tests disabled")
    
    # Skip slow tests if requested
    if item.get_closest_marker("slow"):
        if os.environ.get("SKIP_SLOW_TESTS", "false").lower() == "true":
            pytest.skip("Slow tests disabled")
    
    # Skip performance tests in CI unless specifically requested
    if item.get_closest_marker("performance"):
        if os.environ.get("CI") and not os.environ.get("RUN_PERFORMANCE_TESTS"):
            pytest.skip("Performance tests disabled in CI")

# Custom pytest hooks
def pytest_runtest_call(pyfuncitem):
    """Called to execute the test"""
    # Add any custom test execution logic here
    pass

def pytest_runtest_teardown(item):
    """Called after test execution"""
    # Add any custom teardown logic here
    pass

def pytest_sessionstart(session):
    """Called after session starts"""
    print(f"\nStarting APG Notification System test suite...")
    print(f"Test configuration: {TEST_CONFIG['test_tenant_id']}")

def pytest_sessionfinish(session, exitstatus):
    """Called after session finishes"""
    if exitstatus == 0:
        print(f"\n✅ All tests passed successfully!")
    else:
        print(f"\n❌ Some tests failed (exit status: {exitstatus})")

# Pytest plugins
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov",
    "pytest_xdist"
]