"""
Common test fixtures and configuration for crawler tests.

This module provides shared fixtures, mock objects, and configuration
for all crawler test modules.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.crawlers.base_crawler import BaseCrawler
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def mock_response():
    """Create mock HTTP response."""
    response = Mock()
    response.status_code = 200
    response.text = """
    <html>
        <head>
            <title>Test Article</title>
            <meta name="description" content="Test description">
            <meta name="author" content="Test Author">
        </head>
        <body>
            <h1>Test Headline</h1>
            <p>Test content paragraph 1.</p>
            <p>Test content paragraph 2.</p>
            <img src="test-image.jpg" alt="Test image">
            <a href="test-link.html">Test link</a>
        </body>
    </html>
    """
    response.headers = {
        'Content-Type': 'text/html',
        'Content-Length': str(len(response.text))
    }
    return response


@pytest.fixture
def mock_json_response():
    """Create mock JSON response."""
    response = Mock()
    response.status_code = 200
    response.text = '{"test": "data", "timestamp": "2023-01-01T12:00:00Z"}'
    response.headers = {
        'Content-Type': 'application/json',
        'Content-Length': str(len(response.text))
    }
    return response


@pytest.fixture
def mock_error_response():
    """Create mock error response."""
    response = Mock()
    response.status_code = 404
    response.text = 'Not Found'
    response.headers = {
        'Content-Type': 'text/html',
        'Content-Length': str(len(response.text))
    }
    return response


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    mock_db = Mock(spec=PgSQLManager)
    mock_db.store_article = AsyncMock(return_value={"id": "test-id", "created_at": datetime.now()})
    mock_db.get_article = AsyncMock(return_value=None)
    mock_db.update_article = AsyncMock()
    mock_db.delete_article = AsyncMock()
    mock_db.search_articles = AsyncMock(return_value=[])
    mock_db.is_connected = Mock(return_value=True)
    mock_db.get_connection_info = Mock(return_value={"host": "localhost", "database": "test"})
    return mock_db


@pytest.fixture
def mock_cache_manager(temp_dir):
    """Create mock cache manager."""
    mock_cache = Mock(spec=CacheManager)
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_cache.delete = AsyncMock()
    mock_cache.clear = AsyncMock()
    mock_cache.exists = AsyncMock(return_value=False)
    mock_cache.cache_dir = temp_dir
    return mock_cache


@pytest.fixture
def basic_crawler_config():
    """Basic crawler configuration for testing."""
    return {
        'max_concurrent': 5,
        'retry_count': 3,
        'retry_delay': 1.0,
        'timeout': 30.0,
        'rate_limit': 10,
        'rate_window': 60,
        'stealth_enabled': True,
        'bypass_enabled': True,
        'cache_enabled': True,
        'cache_ttl': 3600
    }


@pytest.fixture
def news_crawler_config():
    """News crawler specific configuration."""
    return {
        'max_concurrent': 5,
        'retry_count': 3,
        'timeout': 30.0,
        'stealth_enabled': True,
        'bypass_enabled': True,
        'cache_enabled': True,
        'cache_ttl': 3600,
        'content_extraction': {
            'extract_text': True,
            'extract_images': True,
            'extract_links': True,
            'extract_metadata': True
        },
        'stealth_config': {
            'use_proxies': False,
            'rotate_user_agents': True,
            'delay_range': [1, 3],
            'respect_robots_txt': True
        }
    }


@pytest.fixture
def performance_config():
    """Performance optimized configuration."""
    return {
        'max_concurrent': 10,
        'retry_count': 2,
        'timeout': 15.0,
        'stealth_enabled': True,
        'cache_enabled': True,
        'rate_limit': 20,
        'rate_window': 1,
        'content_extraction': {
            'extract_text': True,
            'extract_images': False,
            'extract_links': False,
            'extract_metadata': True
        }
    }


@pytest.fixture
def security_config():
    """Security focused configuration."""
    return {
        'max_concurrent': 3,
        'retry_count': 2,
        'timeout': 10.0,
        'stealth_enabled': True,
        'bypass_enabled': False,
        'cache_enabled': True,
        'security_config': {
            'validate_ssl': True,
            'check_content_type': True,
            'max_content_size': 10 * 1024 * 1024,
            'sanitize_content': True,
            'block_dangerous_urls': True
        }
    }


@pytest.fixture
def base_crawler(basic_crawler_config):
    """Create base crawler instance."""
    return BaseCrawler(config=basic_crawler_config)


@pytest.fixture
def news_crawler(news_crawler_config, mock_db_manager, mock_cache_manager):
    """Create news crawler instance."""
    return NewsCrawler(
        config=news_crawler_config,
        db_manager=mock_db_manager,
        cache_manager=mock_cache_manager
    )


@pytest.fixture
def test_urls():
    """Common test URLs."""
    return {
        'valid_urls': [
            'https://httpbin.org/html',
            'https://httpbin.org/json',
            'https://httpbin.org/xml',
            'https://httpbin.org/user-agent',
            'https://httpbin.org/headers'
        ],
        'error_urls': [
            'https://httpbin.org/status/404',
            'https://httpbin.org/status/500',
            'https://httpbin.org/status/503'
        ],
        'slow_urls': [
            'https://httpbin.org/delay/1',
            'https://httpbin.org/delay/2',
            'https://httpbin.org/delay/3'
        ],
        'redirect_urls': [
            'https://httpbin.org/redirect/1',
            'https://httpbin.org/redirect/3',
            'https://httpbin.org/redirect-to?url=https://httpbin.org/json'
        ]
    }


@pytest.fixture
def sample_html_content():
    """Sample HTML content for testing."""
    return {
        'news_article': """
        <html>
            <head>
                <title>Breaking News: Major Event Occurs</title>
                <meta name="description" content="Description of major news event">
                <meta name="author" content="News Reporter">
                <meta name="publish-date" content="2023-01-01T12:00:00Z">
                <meta property="og:image" content="https://example.com/image.jpg">
            </head>
            <body>
                <article>
                    <h1>Breaking News: Major Event Occurs</h1>
                    <p class="lead">This is the lead paragraph of the news article.</p>
                    <div class="content">
                        <p>First paragraph of the main content.</p>
                        <p>Second paragraph with more details.</p>
                        <p>Third paragraph with analysis.</p>
                    </div>
                    <div class="author">By News Reporter</div>
                    <div class="timestamp">January 1, 2023</div>
                    <img src="news-image.jpg" alt="News image">
                    <a href="related-article.html">Related article</a>
                </article>
            </body>
        </html>
        """,
        'blog_post': """
        <html>
            <head>
                <title>Blog Post Title</title>
                <meta name="description" content="Blog post description">
            </head>
            <body>
                <div class="post">
                    <h2>Blog Post Title</h2>
                    <p>Blog post content goes here.</p>
                    <p>More blog content.</p>
                </div>
            </body>
        </html>
        """,
        'minimal_page': """
        <html>
            <head><title>Minimal Page</title></head>
            <body><p>Minimal content</p></body>
        </html>
        """,
        'malformed_html': """
        <html>
            <head>
                <title>Malformed Page
            </head>
            <body>
                <p>Unclosed paragraph
                <div>Nested content</p>
            </body>
        </html>
        """
    }


@pytest.fixture
def malicious_content():
    """Malicious content for security testing."""
    return {
        'xss_attempts': [
            '<script>alert("XSS")</script>',
            '<img src="x" onerror="alert(1)">',
            '<svg onload="alert(1)">',
            'javascript:alert("XSS")',
            '<iframe src="javascript:alert(1)"></iframe>'
        ],
        'sql_injection_attempts': [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; INSERT INTO users VALUES('hacker'); --",
            "' UNION SELECT * FROM passwords --"
        ],
        'malicious_urls': [
            'https://example.com/../../etc/passwd',
            'https://example.com/admin/../../../etc/shadow',
            'file:///etc/passwd',
            'ftp://malicious.com/payload'
        ],
        'oversized_content': 'A' * (20 * 1024 * 1024)  # 20MB
    }


@pytest.fixture
def mock_stealth_orchestrator():
    """Create mock stealth orchestrator."""
    mock_stealth = Mock()
    mock_stealth.fetch_with_stealth = AsyncMock()
    mock_stealth.rotate_user_agent = Mock()
    mock_stealth.get_current_user_agent = Mock(return_value="Mozilla/5.0 (Test Agent)")
    mock_stealth.is_stealth_enabled = Mock(return_value=True)
    return mock_stealth


@pytest.fixture
def mock_bypass_manager():
    """Create mock bypass manager."""
    mock_bypass = Mock()
    mock_bypass.handle_403_error = AsyncMock(return_value=True)
    mock_bypass.handle_rate_limit = AsyncMock(return_value=True)
    mock_bypass.handle_captcha = AsyncMock(return_value=True)
    mock_bypass.is_bypass_enabled = Mock(return_value=True)
    return mock_bypass


@pytest.fixture
def mock_content_parser():
    """Create mock content parser."""
    mock_parser = Mock()
    mock_parser.parse_html = Mock(return_value={
        'title': 'Test Title',
        'content': 'Test content',
        'metadata': {'author': 'Test Author'},
        'images': ['test-image.jpg'],
        'links': ['test-link.html']
    })
    mock_parser.extract_text = Mock(return_value='Test content')
    mock_parser.extract_images = Mock(return_value=['test-image.jpg'])
    mock_parser.extract_links = Mock(return_value=['test-link.html'])
    mock_parser.extract_metadata = Mock(return_value={'author': 'Test Author'})
    return mock_parser


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "functional: mark test as functional test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "usability: mark test as usability test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "external: mark test as requiring external resources"
    )


# Test cleanup
@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test."""
    yield
    # Add any cleanup code here
    pass


# Async test helpers
@pytest.fixture
def async_test_timeout():
    """Default timeout for async tests."""
    return 30.0


@pytest.fixture
def event_loop_policy():
    """Event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


# Performance test helpers
@pytest.fixture
def performance_thresholds():
    """Performance test thresholds."""
    return {
        'max_response_time': 5.0,
        'min_throughput': 5.0,
        'max_memory_usage': 100.0,
        'max_cpu_usage': 80.0
    }


# Security test helpers
@pytest.fixture
def security_test_vectors():
    """Security test vectors."""
    return {
        'safe_urls': [
            'https://httpbin.org/html',
            'https://httpbin.org/json'
        ],
        'unsafe_urls': [
            'javascript:alert(1)',
            'data:text/html,<script>alert(1)</script>'
        ],
        'safe_content': '<p>Safe content</p>',
        'unsafe_content': '<script>alert(1)</script>'
    }


# Utility functions
def create_test_response(content: str, status_code: int = 200, content_type: str = 'text/html') -> Mock:
    """Create a test response object."""
    response = Mock()
    response.status_code = status_code
    response.text = content
    response.headers = {
        'Content-Type': content_type,
        'Content-Length': str(len(content))
    }
    return response


def create_test_article_data(title: str = "Test Title", content: str = "Test content") -> Dict[str, Any]:
    """Create test article data."""
    return {
        'url': 'https://example.com/test',
        'title': title,
        'content': content,
        'metadata': {
            'author': 'Test Author',
            'publish_date': '2023-01-01',
            'description': 'Test description'
        },
        'images': ['test-image.jpg'],
        'links': ['test-link.html'],
        'timestamp': datetime.now(timezone.utc).isoformat()
    }


# Session-level fixtures
@pytest.fixture(scope="session")
def test_database_url():
    """Test database URL."""
    return "postgresql://test:test@localhost/test_db"


@pytest.fixture(scope="session")
def test_cache_dir(tmp_path_factory):
    """Test cache directory."""
    return tmp_path_factory.mktemp("cache")


# Module-level fixtures
@pytest.fixture(scope="module")
def module_test_data():
    """Module-level test data."""
    return {
        'start_time': datetime.now(timezone.utc),
        'test_count': 0
    }


# Class-level fixtures
@pytest.fixture(scope="class")
def class_test_data():
    """Class-level test data."""
    return {
        'test_class': None,
        'setup_complete': False
    }


# Parametrized fixtures
@pytest.fixture(params=[1, 3, 5, 10])
def concurrency_levels(request):
    """Different concurrency levels for testing."""
    return request.param


@pytest.fixture(params=[1, 2, 3, 5])
def retry_counts(request):
    """Different retry counts for testing."""
    return request.param


@pytest.fixture(params=[5, 10, 30, 60])
def timeout_values(request):
    """Different timeout values for testing."""
    return request.param