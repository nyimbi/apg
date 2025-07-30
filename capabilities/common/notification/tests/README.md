# APG Notification System Test Suite

Comprehensive test suite for the APG Notification System, providing thorough coverage of all system components including unit tests, integration tests, performance tests, and security tests.

## 🏗️ Test Architecture

The test suite is organized into several categories:

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and end-to-end workflows
- **Performance Tests**: Validate system performance under load
- **Security Tests**: Validate security controls and compliance
- **End-to-End Tests**: Test complete user workflows

## 📁 Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Global pytest configuration and fixtures
├── fixtures.py              # Shared test fixtures and data
├── utils.py                 # Test utilities and helpers
├── pytest.ini              # Pytest configuration
├── requirements.txt         # Test dependencies
├── README.md               # This file
│
├── test_service.py          # Core notification service tests
├── test_analytics.py        # Analytics engine tests
├── test_security.py         # Security engine tests
├── test_geofencing.py       # Geofencing engine tests
├── test_integration.py      # Integration tests
│
└── ci/                      # CI-specific tests (auto-discovered)
    ├── test_smoke.py        # Smoke tests for CI
    ├── test_api.py          # API contract tests
    └── test_deployment.py   # Deployment validation tests
```

## 🚀 Running Tests

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export TESTING=true
   export TEST_DATABASE_URL="sqlite:///test_notifications.db"
   export TEST_REDIS_URL="redis://localhost:6379/15"
   ```

### Running All Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=../

# Generate HTML coverage report
pytest --cov=../ --cov-report=html
```

### Running Specific Test Categories

```bash
# Run only unit tests
pytest -m "not integration and not performance"

# Run only integration tests
pytest -m integration

# Run only performance tests
pytest -m performance

# Run only security tests
pytest -m security

# Skip slow tests
pytest -m "not slow"

# Skip external service tests
pytest -m "not external"
```

### Running Specific Test Files

```bash
# Run service tests only
pytest test_service.py

# Run analytics tests only
pytest test_analytics.py

# Run security tests only
pytest test_security.py

# Run geofencing tests only
pytest test_geofencing.py

# Run integration tests only
pytest test_integration.py
```

### Running Tests in Parallel

```bash
# Run tests in parallel using pytest-xdist
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## 🧪 Test Categories

### Unit Tests

Test individual components in isolation:

```bash
# Run unit tests for notification service
pytest test_service.py::TestNotificationService

# Run unit tests for analytics engine
pytest test_analytics.py::TestAnalyticsEngine

# Run unit tests for security engine
pytest test_security.py::TestSecurityEngine
```

### Integration Tests

Test component interactions:

```bash
# Run full system integration tests
pytest test_integration.py::TestFullSystemIntegration

# Run external service integration tests
pytest test_integration.py::TestExternalServiceIntegration
```

### Performance Tests

Validate system performance:

```bash
# Run performance tests (may take longer)
pytest -m performance

# Run specific performance test
pytest test_service.py::TestNotificationServicePerformance
```

### Security Tests

Validate security controls:

```bash
# Run security tests
pytest -m security

# Run compliance tests
pytest test_security.py::TestSecurityEngine::test_gdpr_compliance_workflow
```

## 📊 Test Configuration

### Pytest Configuration

The `pytest.ini` file contains:
- Test discovery patterns
- Async test configuration
- Output formatting
- Test markers
- Coverage settings

### Environment Variables

Key environment variables for testing:

```bash
# Core testing
TESTING=true
LOG_LEVEL=DEBUG

# Database
TEST_DATABASE_URL="sqlite:///test_notifications.db"
DATABASE_URL="sqlite:///test_notifications.db"

# Cache
TEST_REDIS_URL="redis://localhost:6379/15"
REDIS_URL="redis://localhost:6379/15"

# External services
SKIP_EXTERNAL_TESTS=false
SKIP_SLOW_TESTS=false
RUN_PERFORMANCE_TESTS=false

# CI/CD
CI=false
GITHUB_ACTIONS=false
```

### Test Data Configuration

Test configuration is defined in `__init__.py`:

```python
TEST_CONFIG = {
    'test_tenant_id': 'test-tenant-12345',
    'test_user_id': 'test-user-67890',
    'redis_url': 'redis://localhost:6379/15',
    'database_url': 'sqlite:///test_notifications.db',
    'mock_external_services': True,
    'log_level': 'DEBUG'
}
```

## 🔧 Test Utilities

### Fixtures

Common fixtures are provided in `fixtures.py`:

- `notification_service`: Mock notification service
- `sample_user_profile`: Test user profile
- `sample_notification_template`: Test template
- `mock_channel_providers`: Mock delivery channels
- `sample_locations`: Test location data

### Utilities

Test utilities in `utils.py`:

- `TestTimer`: Performance timing
- `MockWebhookServer`: Webhook testing
- `TestDataBuilder`: Test data generation
- `AsyncTestRunner`: Async test execution
- `PerformanceTracker`: Performance metrics

### Mock Objects

Comprehensive mock objects for:
- External API services
- Database connections
- Cache clients
- Channel providers
- WebSocket connections

## 📈 Coverage Requirements

The test suite aims for:
- **Minimum 85% overall code coverage**
- **Minimum 90% coverage for core components**
- **100% coverage for security-critical functions**

Generate coverage reports:

```bash
# Generate terminal coverage report
pytest --cov=../

# Generate HTML coverage report
pytest --cov=../ --cov-report=html

# Generate XML coverage report (for CI)
pytest --cov=../ --cov-report=xml
```

## 🔍 Test Debugging

### Debugging Failed Tests

```bash
# Run with detailed output
pytest -vvv

# Stop on first failure
pytest -x

# Run specific test with debugging
pytest test_service.py::test_specific_function -vvv

# Use pdb debugger
pytest --pdb

# Use ipdb debugger (enhanced)
pytest --pdb --pdbcls=IPython.terminal.debugger:TerminalPdb
```

### Logging Configuration

Tests use structured logging:

```python
import logging
logger = logging.getLogger(__name__)

# In tests
logger.debug("Debug information")
logger.info("Test progress")
logger.warning("Warning message")
logger.error("Error occurred")
```

## 🚦 Continuous Integration

### GitHub Actions

The tests integrate with GitHub Actions for CI/CD:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r tests/requirements.txt
      - run: pytest --cov=capabilities/common/notification
```

### Test Reports

The CI generates:
- JUnit XML reports for test results
- Coverage reports in multiple formats
- Performance benchmarks
- Security scan results

## 🎯 Test Markers

Custom pytest markers for test organization:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Unit test"""
    pass

@pytest.mark.integration
def test_integration_workflow():
    """Integration test"""
    pass

@pytest.mark.performance
def test_performance_under_load():
    """Performance test"""
    pass

@pytest.mark.security
def test_security_validation():
    """Security test"""
    pass

@pytest.mark.slow
def test_slow_operation():
    """Long-running test"""
    pass

@pytest.mark.external
def test_external_service():
    """Test requiring external services"""
    pass
```

## 📝 Writing New Tests

### Test Structure

Follow this structure for new tests:

```python
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ..models import *
from .fixtures import *
from .utils import *

class TestNewFeature:
    """Test new feature functionality"""
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self, notification_service):
        """Test basic functionality"""
        # Arrange
        # ... setup test data
        
        # Act
        result = await notification_service.new_feature()
        
        # Assert
        assert result is not None
        assert result.status == 'success'
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_performance(self):
        """Test performance under load"""
        with TestTimer() as timer:
            # ... performance test code
            pass
        
        assert timer.elapsed < 5.0  # Max 5 seconds
```

### Best Practices

1. **Isolation**: Each test should be independent
2. **Clarity**: Test names should clearly describe what's being tested
3. **Coverage**: Test both success and failure cases
4. **Performance**: Include performance assertions where relevant
5. **Mocking**: Mock external dependencies appropriately
6. **Cleanup**: Use fixtures for setup and teardown

## 🐛 Troubleshooting

### Common Issues

1. **Async Test Issues**:
   ```bash
   # Ensure proper async configuration
   pytest_plugins = ["pytest_asyncio"]
   ```

2. **Database Connection Issues**:
   ```bash
   # Check database URL and permissions
   export TEST_DATABASE_URL="sqlite:///test_notifications.db"
   ```

3. **Redis Connection Issues**:
   ```bash
   # Ensure Redis is running and accessible
   redis-cli ping
   ```

4. **Import Issues**:
   ```bash
   # Ensure Python path includes notification directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Performance Issues

If tests are running slowly:

1. Use `pytest -n auto` for parallel execution
2. Skip slow tests with `-m "not slow"`
3. Use `--lf` to run only last failed tests
4. Use `--tb=no` to reduce output

## 📚 Documentation

For more information, see:

- [Main Documentation](../README.md)
- [API Documentation](../docs/api.md)
- [Architecture Documentation](../docs/architecture.md)
- [Security Documentation](../docs/security.md)

## 🤝 Contributing

When contributing tests:

1. Follow the existing test structure
2. Include both positive and negative test cases
3. Add performance tests for new features
4. Update documentation as needed
5. Ensure all tests pass before submitting PR

## 📞 Support

For test-related issues:

1. Check this README first
2. Review existing tests for examples
3. Check the [troubleshooting section](#-troubleshooting)
4. Create an issue with test logs and environment details

---

Happy Testing! 🧪✨