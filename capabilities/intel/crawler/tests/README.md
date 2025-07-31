# Crawler Test Suite

This directory contains comprehensive tests for the crawler components of the Lindela platform.

## Test Structure

The test suite is organized into six main categories:

### 📁 Test Categories

```
tests/
├── unit/           # Unit tests - test individual components in isolation
├── integration/    # Integration tests - test component interactions  
├── functional/     # Functional tests - test complete workflows
├── performance/    # Performance tests - test under load conditions
├── security/       # Security tests - test against vulnerabilities
└── usability/      # Usability tests - test user experience
```

### 🔧 Configuration Files

- `conftest.py` - Shared fixtures and test configuration
- `pytest.ini` - Pytest configuration and markers
- `run_tests.py` - Test runner script with various options
- `README.md` - This file

## Running Tests

### Quick Start

```bash
# Run all tests
python run_tests.py all

# Run specific test category
python run_tests.py unit
python run_tests.py integration
python run_tests.py functional

# Run with coverage
python run_tests.py all --coverage

# Run with verbose output
python run_tests.py unit --verbose
```

### Test Categories

#### Unit Tests
Test individual components in isolation with mocked dependencies.

```bash
python run_tests.py unit
```

**Coverage:**
- `test_base_crawler.py` - Base crawler functionality
- Individual component tests
- Mock-based isolated testing

#### Integration Tests
Test interactions between different components.

```bash
python run_tests.py integration
```

**Coverage:**
- `test_news_crawler_integration.py` - Component interactions
- Database integration
- Cache integration
- Stealth and bypass integration

#### Functional Tests
Test complete workflows from end-to-end.

```bash
python run_tests.py functional
```

**Coverage:**
- `test_news_crawler_functional.py` - Real-world scenarios
- Full crawl workflows
- Error handling scenarios
- Content extraction quality

#### Performance Tests
Test behavior under load conditions.

```bash
python run_tests.py performance
```

**Coverage:**
- `test_news_crawler_performance.py` - Load testing
- Concurrent processing
- Memory usage analysis
- Throughput measurement

#### Security Tests
Test against security vulnerabilities.

```bash
python run_tests.py security
```

**Coverage:**
- `test_news_crawler_security.py` - Security validation
- XSS prevention
- SQL injection prevention
- URL validation
- Content sanitization

#### Usability Tests
Test user experience and API design.

```bash
python run_tests.py usability
```

**Coverage:**
- `test_news_crawler_usability.py` - User experience
- API intuitiveness
- Error message quality
- Configuration simplicity

### Advanced Usage

#### Run Tests by Marker

```bash
# Run slow tests
python run_tests.py --marker slow

# Run external tests
python run_tests.py --marker external

# Run network-dependent tests
python run_tests.py --marker network
```

#### Run Specific Tests

```bash
# Run specific test file
python run_tests.py --test unit/test_base_crawler.py

# Run specific test method
python run_tests.py --test unit/test_base_crawler.py::TestBaseCrawler::test_crawler_initialization
```

#### Quick Test Sets

```bash
# Run quick tests (unit + integration)
python run_tests.py quick

# Run slow tests (performance + functional)
python run_tests.py slow
```

### Using pytest directly

```bash
# Run all tests with coverage
pytest --cov=packages.crawlers --cov-report=html

# Run specific test category
pytest -m unit
pytest -m integration
pytest -m performance

# Run with specific verbosity
pytest -v unit/
pytest -vv integration/

# Run with custom markers
pytest -m "unit and not slow"
pytest -m "integration or functional"
```

## Test Markers

The test suite uses the following markers for categorization:

### Primary Markers
- `unit` - Unit tests
- `integration` - Integration tests
- `functional` - Functional tests
- `performance` - Performance tests
- `security` - Security tests
- `usability` - Usability tests

### Secondary Markers
- `slow` - Slow running tests (>5 seconds)
- `external` - Tests requiring external resources
- `network` - Tests requiring network access
- `database` - Tests requiring database access
- `cache` - Tests requiring cache access
- `mock` - Tests using mocks
- `real` - Tests using real services

### Component Markers
- `stealth` - Tests for stealth capabilities
- `bypass` - Tests for bypass mechanisms
- `parser` - Tests for content parsing
- `crawler` - Tests for crawler functionality

## Test Environment Setup

### Prerequisites

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# For performance tests
pip install psutil

# For security tests (optional)
pip install safety bandit
```

### Environment Check

```bash
# Check if test environment is properly set up
python run_tests.py --check
```

### Test Data

The test suite uses:
- Mock responses for isolated testing
- httpbin.org for functional testing
- Temporary directories for file operations
- In-memory databases for database tests

## Coverage Reports

### Generate Coverage Report

```bash
# Generate HTML coverage report
python run_tests.py --coverage-report

# View coverage report
open htmlcov/index.html
```

### Coverage Targets

- **Unit Tests**: >95% coverage
- **Integration Tests**: >90% coverage
- **Overall**: >85% coverage

## Test Configuration

### Pytest Configuration

The `pytest.ini` file contains:
- Test discovery patterns
- Marker definitions
- Coverage configuration
- Logging configuration
- Asyncio settings

### Shared Fixtures

The `conftest.py` file provides:
- Mock objects for external dependencies
- Test data generators
- Configuration fixtures
- Utility functions

## Continuous Integration

### GitHub Actions Integration

```yaml
# Example CI configuration
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.12
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: python run_tests.py all --coverage
```

### Local Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Ensure PYTHONPATH includes project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Or use development install
pip install -e .
```

#### Async Test Issues
```bash
# Ensure pytest-asyncio is installed
pip install pytest-asyncio

# Check asyncio mode in pytest.ini
asyncio_mode = auto
```

#### Coverage Issues
```bash
# Ensure pytest-cov is installed
pip install pytest-cov

# Check coverage configuration
pytest --cov=packages.crawlers --cov-report=term-missing
```

### Performance Test Issues

Performance tests may fail on slower systems. Adjust thresholds in `conftest.py`:

```python
@pytest.fixture
def performance_thresholds():
    return {
        'max_response_time': 10.0,  # Increased from 5.0
        'min_throughput': 2.0,      # Decreased from 5.0
        'max_memory_usage': 200.0,  # Increased from 100.0
        'max_cpu_usage': 90.0       # Increased from 80.0
    }
```

### External Resource Tests

Some tests require external resources:

```bash
# Skip external tests
pytest -m "not external"

# Run only external tests
pytest -m external
```

## Best Practices

### Writing Tests

1. **Use descriptive test names**
   ```python
   def test_crawler_handles_404_errors_gracefully(self):
   ```

2. **Use appropriate markers**
   ```python
   @pytest.mark.unit
   @pytest.mark.asyncio
   async def test_crawler_initialization(self):
   ```

3. **Use fixtures for setup**
   ```python
   def test_crawler_with_config(self, basic_crawler_config):
   ```

4. **Test edge cases**
   ```python
   def test_crawler_with_empty_config(self):
   def test_crawler_with_invalid_config(self):
   ```

### Test Organization

1. **Group related tests in classes**
   ```python
   class TestBaseCrawler:
       def test_initialization(self):
       def test_configuration(self):
   ```

2. **Use clear test structure**
   ```python
   def test_something(self):
       # Arrange
       crawler = NewsCrawler()
       
       # Act
       result = crawler.do_something()
       
       # Assert
       assert result is not None
   ```

3. **Clean up resources**
   ```python
   async def test_with_cleanup(self):
       async with crawler:
           # test code
       # automatic cleanup
   ```

## Contributing

### Adding New Tests

1. **Choose appropriate category**
   - Unit tests for isolated components
   - Integration tests for component interactions
   - Functional tests for end-to-end workflows

2. **Use existing fixtures**
   - Check `conftest.py` for available fixtures
   - Add new fixtures if needed

3. **Follow naming conventions**
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

4. **Add appropriate markers**
   ```python
   @pytest.mark.unit
   @pytest.mark.asyncio
   async def test_new_feature(self):
   ```

### Test Documentation

1. **Document test purpose**
   ```python
   def test_specific_behavior(self):
       """Test that specific behavior works correctly under specific conditions."""
   ```

2. **Document complex setups**
   ```python
   @pytest.fixture
   def complex_setup():
       """Create complex test setup with multiple components."""
   ```

3. **Update this README**
   - Add new test categories
   - Document new markers
   - Update usage examples

## Support

For questions or issues with the test suite:

1. Check this README for common solutions
2. Review test output for specific error messages
3. Check test logs in `tests.log`
4. Review coverage reports for missing test areas

## License

This test suite is part of the Lindela platform and follows the same license terms.