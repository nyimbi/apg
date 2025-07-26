# APG Budgeting & Forecasting - Test Suite

Comprehensive test suite for the APG Budgeting & Forecasting capability covering all features from core budget management to advanced AI-powered analytics.

## Test Structure

```
tests/
├── README.md                    # This file
├── conftest.py                  # Pytest configuration and shared fixtures
├── pytest.ini                  # Pytest settings
├── run_tests.py                 # Test runner script
├── test_integration.py          # Comprehensive integration tests
├── test_advanced_features.py    # Advanced features tests
└── coverage_html/              # Coverage reports (generated)
```

## Test Categories

### Unit Tests (`-m unit`)
- Test individual functions and methods in isolation
- Fast execution, no external dependencies
- Mock external services and databases

### Integration Tests (`-m integration`)
- Test complete workflows and service interactions
- Test APG platform integrations
- Require database connectivity

### Performance Tests (`-m performance`)
- Load testing with large datasets
- Concurrent operation testing
- Performance benchmarking

### Smoke Tests (`-m smoke`)
- Basic functionality verification
- Quick health checks
- Minimal test coverage for CI/CD

## Running Tests

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov pytest-httpserver responses

# Setup test database (optional, uses mocks if not available)
export TEST_DATABASE_URL="postgresql://test_user:test_pass@localhost:5432/test_apg_bf"
```

### Quick Start

```bash
# Run all tests with coverage
python tests/run_tests.py --all

# Run only unit tests
python tests/run_tests.py --unit

# Run integration tests
python tests/run_tests.py --integration

# Run with verbose output
python tests/run_tests.py --all --verbose

# Run tests in parallel (faster)
python tests/run_tests.py --all --parallel

# Run specific test
python tests/run_tests.py --specific test_integration.py::TestCoreBudgetManagement
```

### Using Pytest Directly

```bash
# Run all tests
pytest tests/ -v

# Run tests by marker
pytest -m unit tests/
pytest -m integration tests/
pytest -m "not slow" tests/

# Run with coverage
pytest --cov=budgeting_forecasting --cov-report=html tests/

# Run specific test class
pytest tests/test_integration.py::TestCoreBudgetManagement -v

# Run tests matching pattern
pytest -k "test_budget" tests/
```

## Test Features Covered

### Core Functionality
- ✅ Budget creation, update, deletion
- ✅ Budget line management
- ✅ Template system integration
- ✅ Multi-tenant operations
- ✅ Data validation and error handling

### Advanced Features
- ✅ Real-time collaboration
- ✅ Approval workflows with escalation
- ✅ Version control and audit trails
- ✅ Advanced analytics and reporting
- ✅ Interactive dashboards
- ✅ Custom report builder

### AI-Powered Features
- ✅ ML forecasting engine
- ✅ AI budget recommendations
- ✅ Automated monitoring and alerts
- ✅ Anomaly detection
- ✅ Predictive analytics

### Integration Points
- ✅ APG platform capabilities integration
- ✅ Database operations
- ✅ External API integrations
- ✅ Cross-service communication
- ✅ Error handling and recovery

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)
- `base_tenant_context`: Basic tenant context for testing
- `test_service_config`: Test configuration with mocked services
- `capability_instance`: Full capability instance for integration tests
- `sample_budget_comprehensive`: Complete budget data with all features
- `mock_*_service`: Mocked external services for unit tests

### Sample Data
- Budget templates with various complexity levels
- Multi-tenant test scenarios
- Large datasets for performance testing
- Error scenarios for negative testing

## Coverage Reports

After running tests with coverage:

```bash
# View HTML coverage report
open tests/coverage_html/index.html

# View coverage summary
pytest --cov=budgeting_forecasting --cov-report=term-missing tests/
```

## Performance Testing

### Load Testing
```bash
# Run performance tests
python tests/run_tests.py --performance

# Test specific performance scenarios
pytest -m performance tests/test_integration.py::TestPerformanceAndLoad
```

### Benchmarking
- Concurrent budget operations (5+ simultaneous)
- Large dataset analytics (1000+ budget lines)
- Real-time collaboration with multiple users
- ML model training and prediction performance

## Continuous Integration

### GitHub Actions Example
```yaml
name: APG Budgeting & Forecasting Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_pass
          POSTGRES_USER: test_user
          POSTGRES_DB: test_apg_budgeting_forecasting
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov pytest-httpserver responses
    
    - name: Run tests
      run: python tests/run_tests.py --all --no-coverage
      env:
        TEST_DATABASE_URL: postgresql://test_user:test_pass@localhost:5432/test_apg_budgeting_forecasting
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check database connectivity
   python tests/run_tests.py --check-db
   ```

2. **Missing Dependencies**
   ```bash
   # Check test dependencies
   python tests/run_tests.py --check-deps
   ```

3. **Async Test Issues**
   - Ensure `pytest-asyncio` is installed
   - Use `async def` for async test functions
   - No `@pytest.mark.asyncio` decorator needed (auto mode)

4. **Import Errors**
   - Ensure capability directory is in Python path
   - Check relative imports in test files

### Debug Mode
```bash
# Run tests with debugging
pytest --pdb tests/test_integration.py::specific_test

# Capture stdout/stderr
pytest -s tests/
```

## Contributing to Tests

### Adding New Tests
1. Follow existing naming conventions (`test_*.py`)
2. Use appropriate markers (`@pytest.mark.unit`, etc.)
3. Include docstrings explaining test purpose
4. Use shared fixtures from `conftest.py`
5. Add performance tests for new features

### Test Guidelines
- **Isolation**: Tests should not depend on each other
- **Deterministic**: Tests should produce consistent results
- **Fast**: Unit tests should complete quickly (<1s each)
- **Clear**: Test names and assertions should be self-explanatory
- **Comprehensive**: Cover happy path, edge cases, and error conditions

## Test Metrics

Target coverage and performance metrics:

- **Code Coverage**: >90% for all core modules
- **Test Execution Time**: 
  - Unit tests: <30 seconds total
  - Integration tests: <5 minutes total
  - Performance tests: <10 minutes total
- **Test Success Rate**: >99% on stable branches

---

© 2025 Datacraft. All rights reserved.
For support: nyimbi@gmail.com | www.datacraft.co.ke