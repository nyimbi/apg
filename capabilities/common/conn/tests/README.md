# APG Connection Management - Test Suite

**Author:** Nyimbi Odero
**Company:** Datacraft
**Copyright:** © 2025

## Overview

This directory contains a comprehensive test suite for the APG Connection Management capability. The tests ensure reliability, performance, and correctness of all components including services, data lineage, views, and models.

## Test Structure

```
tests/
├── ci/                          # CI/CD ready tests
│   ├── conftest.py             # Test fixtures and configuration
│   ├── test_service.py         # Service layer tests
│   ├── test_lineage_engine.py  # Data lineage tests
│   ├── test_service_bridge.py  # Async/sync bridge tests
│   ├── test_views.py           # Flask-AppBuilder view tests
│   ├── test_models.py          # SQLAlchemy model tests
│   └── __init__.py
├── coverage/                   # Coverage reports (generated)
├── reports/                    # Test reports (generated)
├── requirements.txt            # Test dependencies
├── run_tests.py               # Test runner script
└── README.md                  # This file
```

## Quick Start

### 1. Install Dependencies

```bash
# Install test requirements
pip install -r tests/requirements.txt

# Or use the test runner
python tests/run_tests.py --install-deps
```

### 2. Run All Tests

```bash
# Basic test execution
python tests/run_tests.py

# With coverage report
python tests/run_tests.py --coverage

# Fast execution (stop on first failure)
python tests/run_tests.py --fast
```

### 3. Run Specific Test Types

```bash
# Unit tests only
python tests/run_tests.py --unit

# Integration tests only
python tests/run_tests.py --integration

# Performance tests only
python tests/run_tests.py --perf
```

## Test Categories

### Unit Tests

Tests individual components in isolation with mocked dependencies.

**Coverage:**
- ✅ Service layer methods (ConnectionManager, FlowExecutor, IntelligentConnector)
- ✅ Data lineage engine algorithms
- ✅ Service bridge async/sync integration
- ✅ Model validation and relationships
- ✅ View logic and form handling

**Example:**
```bash
python tests/run_tests.py --unit
```

### Integration Tests

Tests component interactions with real or realistic dependencies.

**Coverage:**
- ✅ Service to database integration
- ✅ Lineage engine with database persistence
- ✅ Service bridge with real async operations
- ✅ Flask views with service integration
- ✅ End-to-end workflow testing

**Example:**
```bash
python tests/run_tests.py --integration
```

### Performance Tests

Tests system performance under load and stress conditions.

**Coverage:**
- ✅ Concurrent connection handling
- ✅ Large schema discovery performance
- ✅ Lineage graph traversal with large datasets
- ✅ Memory usage and leak detection
- ✅ API response times

**Example:**
```bash
python tests/run_tests.py --perf
```

## Detailed Test Coverage

### 1. Service Layer Tests (`test_service.py`)

**ConnectionManager Tests:**
- ✅ Connection creation, update, deletion
- ✅ Connection testing and validation
- ✅ Health monitoring and metrics
- ✅ Schema discovery with Singer.io
- ✅ Performance metrics collection
- ✅ Error handling and recovery

**FlowExecutor Tests:**
- ✅ Data flow creation and execution
- ✅ Flow validation and scheduling
- ✅ Execution history and logging
- ✅ Flow stopping and management
- ✅ Transformation processing

**IntelligentConnector Tests:**
- ✅ AI-powered field mapping suggestions
- ✅ Performance prediction algorithms
- ✅ Batch size optimization
- ✅ Schema drift detection
- ✅ Data quality rule generation

### 2. Lineage Engine Tests (`test_lineage_engine.py`)

**Core Engine Tests:**
- ✅ Graph construction and management
- ✅ Node and edge creation
- ✅ Sensitive data classification
- ✅ Schema discovery integration
- ✅ Visualization data generation

**Graph Algorithms:**
- ✅ Upstream dependency traversal
- ✅ Downstream impact analysis
- ✅ Path finding and relationship mapping
- ✅ Graph performance with large datasets

**Transformation Tracking:**
- ✅ Filter transformation lineage
- ✅ Aggregation relationship tracking
- ✅ Join operation lineage
- ✅ Field mapping preservation

### 3. Service Bridge Tests (`test_service_bridge.py`)

**Async/Sync Integration:**
- ✅ Event loop management
- ✅ Async operation execution in sync context
- ✅ Error handling and timeouts
- ✅ Concurrent operation handling

**Service Integration:**
- ✅ Connection management bridging
- ✅ Flow execution bridging
- ✅ Lineage operation bridging
- ✅ AI service integration

**Decorator Testing:**
- ✅ Service bridge injection
- ✅ Function metadata preservation
- ✅ Error handling in decorated functions

### 4. View Tests (`test_views.py`)

**Flask-AppBuilder Integration:**
- ✅ CRUD operations for connections
- ✅ Data flow management views
- ✅ Dashboard and analytics views
- ✅ API endpoint functionality

**UI Component Tests:**
- ✅ Form validation and processing
- ✅ Template rendering
- ✅ JavaScript integration
- ✅ User interaction workflows

**API Endpoint Tests:**
- ✅ RESTful API compliance
- ✅ JSON response formatting
- ✅ Error handling and status codes
- ✅ Authentication and authorization

### 5. Model Tests (`test_models.py`)

**Database Models:**
- ✅ Model creation and validation
- ✅ Field constraints and defaults
- ✅ Unique constraints and indexes
- ✅ JSON field storage and retrieval

**Relationships:**
- ✅ Foreign key relationships
- ✅ One-to-many associations
- ✅ Many-to-many relationships
- ✅ Cascade delete behavior

**Data Integrity:**
- ✅ Database constraints
- ✅ Transaction handling
- ✅ Concurrent access safety
- ✅ Data migration compatibility

## Test Configuration

### Environment Variables

The test suite recognizes several environment variables:

```bash
export APG_TEST_MODE=true           # Enable test mode
export APG_LOG_LEVEL=WARNING        # Set log level for tests
export APG_TEST_DB_URL=sqlite://    # Test database URL
export APG_DISABLE_EXTERNAL=true    # Disable external API calls
```

### Database Configuration

Tests use an in-memory SQLite database by default for speed and isolation:

```python
# In conftest.py
@pytest.fixture
def db_engine():
    engine = create_engine("sqlite:///:memory:", ...)
    yield engine
```

For integration tests requiring PostgreSQL features, use:

```bash
pytest -k integration --postgresql
```

### Async Testing

The test suite properly handles async operations:

```python
# Automatic event loop management
@pytest.fixture
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## Test Fixtures

### Core Fixtures

**Database Fixtures:**
- `db_engine` - In-memory SQLite engine
- `db_session` - Database session with automatic cleanup
- `sample_connection` - Pre-created test connection
- `sample_flow` - Pre-created test data flow
- `sample_lineage_nodes` - Test lineage graph nodes
- `sample_lineage_edges` - Test lineage relationships

**Service Fixtures:**
- `connection_manager` - Initialized ConnectionManager
- `service_bridge` - ServiceBridge with mocked dependencies
- `lineage_engine` - DataLineageEngine instance
- `mock_flask_app` - Flask app for view testing

**Mock Data Fixtures:**
- `sample_connection_data` - Connection configuration
- `sample_flow_data` - Flow configuration
- `mock_singer_discovery` - Singer.io catalog response
- `mock_performance_metrics` - System performance data
- `mock_ai_suggestions` - AI service responses

### Custom Assertions

Utility functions for complex assertions:

```python
def assert_connection_data(connection, expected_data):
    """Assert connection matches expected structure"""

def assert_lineage_structure(nodes, edges, expected_structure):
    """Assert lineage graph matches expected format"""
```

## Running Specific Tests

### Individual Test Files

```bash
# Run only service tests
pytest tests/ci/test_service.py -v

# Run only lineage engine tests
pytest tests/ci/test_lineage_engine.py -v

# Run specific test class
pytest tests/ci/test_service.py::TestConnectionManager -v

# Run specific test method
pytest tests/ci/test_service.py::TestConnectionManager::test_create_connection_success -v
```

### Test Patterns

```bash
# Run tests matching pattern
pytest -k "connection and create" -v

# Run tests NOT matching pattern
pytest -k "not performance" -v

# Run tests with specific markers
pytest -m "slow" -v
```

## Coverage Reports

### Generate Coverage

```bash
# HTML coverage report
python tests/run_tests.py --coverage

# View coverage in browser
open tests/coverage/index.html
```

### Coverage Targets

The test suite maintains high coverage standards:

- **Overall Coverage:** > 90%
- **Service Layer:** > 95%
- **Models:** > 90%
- **Views:** > 85%
- **Critical Paths:** 100%

### Coverage Configuration

```ini
# .coveragerc
[run]
source = .
omit =
    tests/*
    */migrations/*
    */venv/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
```

## Continuous Integration

### GitHub Actions Integration

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", 3.11]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r tests/requirements.txt
      - name: Run tests
        run: python tests/run_tests.py --coverage --parallel
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: python tests/run_tests.py --fast
        language: system
        always_run: true
```

## Performance Benchmarking

### Benchmark Tests

Performance tests include benchmarking:

```python
def test_connection_creation_performance(benchmark, connection_manager):
    """Benchmark connection creation performance"""
    result = benchmark(
        connection_manager.create_connection,
        sample_connection_data
    )
    assert result is not None
```

### Performance Targets

- Connection creation: < 100ms
- Flow execution startup: < 500ms
- Lineage graph generation: < 2s for 1000+ nodes
- API response time: < 200ms for simple queries
- Database queries: < 50ms for indexed lookups

## Debugging Tests

### Verbose Output

```bash
# Maximum verbosity
pytest -vvv --tb=long

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb
```

### Log Analysis

```bash
# Enable debug logging
APG_LOG_LEVEL=DEBUG python tests/run_tests.py

# Specific logger output
pytest --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
```

### Memory Profiling

```bash
# Profile memory usage
python -m pytest --memray tests/ci/test_lineage_engine.py::test_large_graph_performance
```

## Contributing

### Adding New Tests

1. **Follow naming conventions:**
   - Test files: `test_*.py`
   - Test classes: `TestClassName`
   - Test methods: `test_method_description`

2. **Use appropriate fixtures:**
   - Use existing fixtures when possible
   - Create specific fixtures for complex scenarios
   - Clean up resources in fixtures

3. **Write clear assertions:**
   - Use descriptive assertion messages
   - Test both positive and negative cases
   - Include edge cases and error conditions

4. **Document complex tests:**
   - Add docstrings for complex test methods
   - Explain test scenarios and expected outcomes
   - Include references to requirements or issues

### Test Quality Standards

- **Isolation:** Tests must not depend on external services
- **Repeatability:** Tests must produce consistent results
- **Speed:** Unit tests should complete in < 1s each
- **Clarity:** Test intent should be obvious from the name
- **Coverage:** New code must include corresponding tests

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure Python path is set
export PYTHONPATH=/path/to/apg/capabilities/common/conn:$PYTHONPATH
```

**Database Errors:**
```bash
# Reset test database
rm -f tests/test.db
python tests/run_tests.py --check-env
```

**Async Test Issues:**
```bash
# Check event loop configuration
pytest tests/ci/test_service_bridge.py::test_real_async_integration -v
```

**Performance Test Timeouts:**
```bash
# Increase timeout for slow systems
pytest --timeout=300 -k performance
```

### Getting Help

1. Check test logs in `tests/reports/`
2. Run with `--debug` flag for detailed output
3. Use `--pdb` to debug failing tests
4. Review fixture configurations in `conftest.py`

## Test Metrics

The test suite generates comprehensive metrics:

- **Test Count:** 100+ tests across all components
- **Execution Time:** < 60 seconds for full suite
- **Code Coverage:** > 90% overall
- **Mock Coverage:** 95% of external dependencies mocked
- **Edge Case Coverage:** Critical error paths tested

---

*For more information about the APG Connection Management capability, see the main [USER_GUIDE.md](../USER_GUIDE.md).*