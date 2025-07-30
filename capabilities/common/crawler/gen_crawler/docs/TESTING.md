# Gen Crawler Testing Guide

Comprehensive testing documentation for the gen_crawler package including test execution, coverage analysis, and testing best practices.

## 🧪 Test Suite Overview

The gen_crawler test suite provides comprehensive coverage of all package components:

### Test Categories

- **Unit Tests**: Core component testing in isolation
- **Integration Tests**: Component interaction testing
- **CLI Tests**: Command-line interface testing
- **Real Site Tests**: Production scenario validation
- **Performance Tests**: Scalability and resource usage testing

### Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_core.py             # Core crawler component tests
├── test_config.py           # Configuration system tests
├── test_parsers.py          # Content parser tests
├── test_cli.py              # CLI interface tests
├── test_integration.py      # Integration tests
├── test_real_sites.py       # Real website testing
└── run_tests.py             # Test runner script
```

## 🚀 Running Tests

### Basic Test Execution

```bash
# Run all tests
python tests/run_tests.py

# Run with verbose output
python tests/run_tests.py --verbose

# Check dependencies
python tests/run_tests.py --deps
```

### Targeted Testing

```bash
# Run only unit tests
python tests/run_tests.py --unit

# Run only integration tests
python tests/run_tests.py --integration

# Run specific module tests
python tests/run_tests.py --module core
python tests/run_tests.py --module config
python tests/run_tests.py --module parsers
python tests/run_tests.py --module cli

# Run specific test method
python tests/run_tests.py --test TestGenCrawler.test_crawler_creation
```

### Coverage Analysis

```bash
# Run tests with coverage analysis
python tests/run_tests.py --coverage

# Install coverage tool if needed
pip install coverage
```

## 📊 Test Components

### Core Tests (`test_core.py`)

Tests for the main crawler components:

- **GenCrawlResult**: Result data structure testing
- **GenSiteResult**: Site result aggregation testing
- **SiteProfile**: Site profiling and performance tracking
- **AdaptiveCrawler**: Strategy management testing
- **GenCrawler**: Main crawler functionality

```python
# Example core test
class TestGenCrawler:
    def test_crawler_creation(self, config):
        crawler = GenCrawler(config)
        assert crawler.config['max_pages_per_site'] == 10
        assert isinstance(crawler.visited_urls, set)
```

### Configuration Tests (`test_config.py`)

Tests for the configuration system:

- **Configuration Data Classes**: Settings validation
- **File Operations**: Save/load configuration files
- **Environment Variables**: Environment integration
- **Factory Functions**: Configuration creation helpers

```python
# Example config test
def test_config_validation(self):
    invalid_settings = GenCrawlerSettings()
    invalid_settings.performance.max_concurrent = -1
    
    with self.assertRaises(ValueError):
        GenCrawlerConfig(settings=invalid_settings)
```

### Parser Tests (`test_parsers.py`)

Tests for content parsing and analysis:

- **Content Extraction**: Multi-method content extraction
- **Quality Analysis**: Content quality scoring
- **Content Classification**: Type detection and categorization
- **Error Handling**: Malformed HTML handling

```python
# Example parser test
def test_content_quality_scoring(self):
    high_quality_content = ParsedSiteContent(
        url="https://example.com/article",
        title="Comprehensive Analysis",
        content="Detailed content...",
        word_count=800,
        authors=["Dr. Jane Smith"]
    )
    
    score = self.analyzer.calculate_quality_score(high_quality_content)
    self.assertGreater(score, 0.7)
```

### CLI Tests (`test_cli.py`)

Tests for the command-line interface:

- **Argument Parsing**: Command and option parsing
- **Command Execution**: Command implementation testing
- **Configuration Building**: CLI to config translation
- **Export Functions**: Output format testing

```python
# Example CLI test
def test_crawl_command_parsing(self):
    args = self.parser.parse_args([
        'crawl', 'https://example.com',
        '--max-pages', '100',
        '--format', 'markdown'
    ])
    
    self.assertEqual(args.command, 'crawl')
    self.assertEqual(args.max_pages, 100)
```

### Integration Tests (`test_integration.py`)

Tests for component interactions:

- **Full Workflow**: End-to-end testing
- **Export Integration**: Export system testing
- **Performance Integration**: Statistics and monitoring
- **Real-world Scenarios**: Use case validation

```python
# Example integration test
async def test_full_crawl_workflow(self):
    config = create_gen_config()
    crawler = GenCrawler(config.get_crawler_config())
    
    result = await crawler.crawl_site("https://example.com")
    assert result.total_pages > 0
    assert result.success_rate > 0
```

## 🌐 Real Site Testing

### Safety Measures

Real site testing uses conservative settings:

```python
# Conservative configuration for real testing
config.settings.performance.max_pages_per_site = 3
config.settings.performance.max_concurrent = 1
config.settings.performance.crawl_delay = 3.0
config.settings.stealth.respect_robots_txt = True
```

### Safe Test Sites

Only approved test sites are used:

- `https://httpbin.org` - HTTP testing service
- `https://example.com` - Standard example domain
- `https://jsonplaceholder.typicode.com` - JSON testing API

### Running Real Site Tests

```bash
# Run real site tests (requires network)
python tests/test_real_sites.py

# Skip real network tests
export SKIP_REAL_NETWORK_TESTS=1
python tests/run_tests.py
```

## 📈 Performance Testing

### Memory Usage Testing

```python
def test_memory_usage_patterns(self):
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Create multiple crawler instances
    crawlers = [GenCrawler(config) for _ in range(5)]
    
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    # Should be reasonable
    self.assertLess(memory_increase, 100)  # Less than 100MB
```

### Configuration Performance

```python
def test_configuration_performance(self):
    start_time = time.time()
    configs = [create_gen_config() for _ in range(100)]
    creation_time = time.time() - start_time
    
    # Should be fast
    self.assertLess(creation_time, 1.0)  # Less than 1 second
```

## 🔧 Mocking and Test Fixtures

### Crawlee Mocking

Since Crawlee requires complex setup, tests use mocking:

```python
@patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True)
@patch('gen_crawler.core.gen_crawler.AdaptivePlaywrightCrawler')
def test_crawler_initialization(self, mock_crawler_class):
    mock_crawler = MagicMock()
    mock_crawler_class.return_value = mock_crawler
    
    crawler = GenCrawler(config)
    await crawler.initialize()
    
    mock_crawler_class.assert_called_once()
```

### Test Data

Realistic test data for validation:

```python
sample_html = """
<html>
<head>
    <title>Test Article: Important News</title>
    <meta name="author" content="Jane Reporter">
</head>
<body>
    <article>
        <h1>Test Article: Important News</h1>
        <p>First paragraph with substantial content.</p>
        <p>Second paragraph with more information.</p>
    </article>
</body>
</html>
"""
```

## 📋 Test Dependencies

### Required Dependencies

- **unittest**: Built-in Python testing framework
- **pytest**: Advanced testing framework (optional)
- **mock**: Mocking library (built-in from Python 3.3+)

### Optional Dependencies

- **coverage**: Code coverage analysis
- **psutil**: System resource monitoring
- **pytest-asyncio**: Async test support

### Installation

```bash
# Install testing dependencies
pip install pytest pytest-asyncio coverage psutil

# Or install development dependencies
pip install -r requirements-dev.txt
```

## 🏃‍♂️ Test Execution Examples

### Development Testing

```bash
# Quick unit tests during development
python tests/run_tests.py --unit --verbose

# Test specific component you're working on
python tests/run_tests.py --module parsers --verbose

# Run with coverage to check test completeness
python tests/run_tests.py --coverage
```

### CI/CD Pipeline Testing

```bash
# Complete test suite for CI
python tests/run_tests.py

# Skip real network tests in CI
export SKIP_REAL_NETWORK_TESTS=1
python tests/run_tests.py --coverage
```

### Pre-release Testing

```bash
# Comprehensive testing before release
python tests/run_tests.py --coverage --verbose

# Include real site testing
unset SKIP_REAL_NETWORK_TESTS
python tests/test_real_sites.py
```

## 📊 Coverage Targets

### Current Coverage Goals

- **Core Components**: > 90%
- **Configuration**: > 95%
- **Parsers**: > 85%
- **CLI**: > 80%
- **Overall**: > 85%

### Coverage Report Example

```
Module                     Coverage
------------------------   --------
gen_crawler.core           92.3%
gen_crawler.config         96.1%
gen_crawler.parsers        88.7%
gen_crawler.cli            82.4%
------------------------   --------
Total Coverage             89.2%
```

## 🐛 Debugging Tests

### Verbose Test Output

```bash
# Maximum verbosity
python tests/run_tests.py --verbose

# Debug specific test
python tests/run_tests.py --test TestGenCrawler.test_crawler_creation --verbose
```

### Test Debugging Tips

1. **Use Print Statements**: Add debug prints in test methods
2. **Check Mock Calls**: Verify mock object interactions
3. **Isolate Tests**: Run individual tests to isolate issues
4. **Check Dependencies**: Ensure all required packages are installed

### Common Test Issues

- **Import Errors**: Check package installation and paths
- **Mock Failures**: Verify mock setup and expectations
- **Async Issues**: Ensure proper async/await usage
- **Network Timeouts**: Skip real network tests in unstable environments

## 📝 Writing New Tests

### Test Naming Convention

```python
# Test class naming
class TestComponentName(unittest.TestCase):

# Test method naming
def test_specific_functionality(self):
def test_error_handling_scenario(self):
def test_edge_case_validation(self):
```

### Test Structure

```python
def test_example_functionality(self):
    # Arrange - Set up test data
    config = create_test_config()
    crawler = GenCrawler(config)
    
    # Act - Execute the functionality
    result = crawler.some_method()
    
    # Assert - Verify expectations
    self.assertIsNotNone(result)
    self.assertEqual(result.status, 'success')
```

### Async Test Example

```python
@pytest.mark.asyncio
async def test_async_functionality(self):
    crawler = GenCrawler(config)
    await crawler.initialize()
    
    result = await crawler.crawl_site("https://example.com")
    
    assert result.total_pages > 0
```

## 🎯 Test Quality Guidelines

### Good Test Practices

1. **Single Responsibility**: One concept per test
2. **Clear Naming**: Descriptive test names
3. **Independent Tests**: No test dependencies
4. **Fast Execution**: Minimize external dependencies
5. **Comprehensive Coverage**: Test happy path and edge cases

### Test Documentation

```python
def test_url_validation_edge_cases(self):
    """
    Test URL validation with various edge cases including:
    - Malformed URLs
    - Missing protocols
    - Invalid characters
    - Empty strings
    """
    # Test implementation...
```

This testing guide ensures comprehensive validation of the gen_crawler package while maintaining development efficiency and code quality.