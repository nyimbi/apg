# APG DVRL Test Suite

## Test Organization

The DVRL test suite is organized into the following categories:

### `/ci/` - Continuous Integration Tests
Tests that run automatically in CI/CD pipeline:
- `test_adapters.py` - Data source adapter tests
- `test_connectors.py` - Database connector tests  
- `test_integration.py` - Core integration tests
- `test_singer_integration.py` - Singer.io integration tests

### `/unit/` - Unit Tests
Focused tests for individual components:
- Test individual functions and classes in isolation
- Use mocks for external dependencies
- Fast execution (<1s per test)
- High code coverage target (>95%)

### `/integration/` - Integration Tests  
Tests that verify component interactions:
- APG platform service integrations
- Database connectivity tests
- End-to-end query execution
- Multi-tenant functionality

### `/performance/` - Performance Tests
Benchmarks and performance validation:
- Load testing scenarios
- Concurrent query execution
- Memory usage validation
- Response time benchmarks

### `/security/` - Security Tests
Security validation and penetration tests:
- Authentication and authorization
- Data masking and privacy
- SQL injection prevention
- Access control validation

## Running Tests

### All Tests
```bash
# Run complete test suite
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=dvrl --cov-report=html
```

### By Category
```bash
# CI tests only (fast)
uv run pytest tests/ci/ -v

# Unit tests
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# Performance benchmarks
uv run pytest tests/performance/ -v --benchmark-only

# Security tests
uv run pytest tests/security/ -v
```

### Specific Test Files
```bash
# Run specific test file
uv run pytest tests/ci/test_adapters.py -v

# Run specific test method
uv run pytest tests/unit/test_service.py::TestDVRLService::test_query_parsing -v
```

## Test Configuration

### Environment Variables
```bash
export DVRL_TEST_MODE="true"
export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/dvrl_test"
export TEST_REDIS_URL="redis://localhost:6379/1"
export APG_TEST_BASE_URL="http://localhost:8080"
```

### Fixtures and Setup
Common test fixtures are defined in `conftest.py`:
- Database setup/teardown
- Mock APG services
- Sample data creation
- Authentication tokens

## Writing Tests

### Test Standards
- Follow pytest conventions
- Use descriptive test names
- Include docstrings for complex tests
- Use appropriate fixtures
- Clean up resources in teardown

### Example Test Structure
```python
import pytest
import asyncio
from dvrl.service import DVRLService

class TestDVRLService:
    \"\"\"Test DVRL service functionality\"\"\"
    
    @pytest.fixture
    async def service(self):
        service = DVRLService(tenant_id="test", user_id="test")
        await service.initialize()
        yield service
        await service.cleanup()
    
    async def test_query_execution(self, service):
        \"\"\"Test basic query execution\"\"\"
        result = await service.execute_federated_query("SELECT 1")
        assert result.status == "completed"
        assert result.rows_returned == 1
```

## Test Data Management

### Sample Data
- Use consistent test data across tests
- Create realistic but minimal datasets
- Include edge cases and error conditions
- Clean up test data after each test

### Mock Services
- Mock external APG services for unit tests
- Use real services for integration tests
- Provide consistent mock responses
- Document mock behavior and expectations

---

**Version**: 1.0  
**Last Updated**: 2025-01-11  
**Author**: APG Platform Team