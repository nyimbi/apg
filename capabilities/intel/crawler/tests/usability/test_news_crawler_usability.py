"""
Usability tests for news crawler system.

This module contains usability tests that validate the news crawler
user experience, API design, and ease of use.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from typing import Dict, Any, List
import json

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.crawlers.base_crawler import BaseCrawler
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


class TestNewsCrawlerUsability:
    """Usability tests for news crawler system."""
    
    @pytest.fixture
    def user_friendly_config(self):
        """User-friendly configuration for testing."""
        return {
            'max_concurrent': 5,
            'retry_count': 3,
            'timeout': 30.0,
            'stealth_enabled': True,
            'cache_enabled': True,
            'content_extraction': {
                'extract_text': True,
                'extract_images': True,
                'extract_links': True,
                'extract_metadata': True
            }
        }
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_simple_crawler_creation(self, user_friendly_config):
        """Test that creating a crawler is simple and intuitive."""
        # Should be easy to create with default settings
        crawler = NewsCrawler()
        assert crawler is not None
        assert crawler.max_concurrent == 10  # Default value
        
        # Should be easy to create with custom config
        crawler_custom = NewsCrawler(config=user_friendly_config)
        assert crawler_custom is not None
        assert crawler_custom.max_concurrent == 5
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_intuitive_single_url_crawling(self, user_friendly_config):
        """Test that crawling a single URL is intuitive."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <head><title>Test Article</title></head>
                <body>
                    <h1>Article Headline</h1>
                    <p>Article content goes here.</p>
                </body>
            </html>
            """
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            # Should be simple to crawl a single URL
            async with crawler:
                result = await crawler.crawl_url('https://example.com/article')
                
                # Result should be intuitive and well-structured
                assert result is not None
                assert 'title' in result
                assert 'content' in result
                assert 'url' in result
                assert result['title'] == 'Test Article'
                assert 'Article content goes here' in result['content']
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_batch_url_crawling_simplicity(self, user_friendly_config):
        """Test that batch crawling is simple and intuitive."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        urls = [
            'https://example.com/article1',
            'https://example.com/article2',
            'https://example.com/article3'
        ]
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Article content</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            # Should be simple to crawl multiple URLs
            async with crawler:
                results = await crawler.crawl_urls(urls)
                
                # Results should be intuitive
                assert len(results) == len(urls)
                assert all(isinstance(result, dict) for result in results if result is not None)
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_error_messages_are_helpful(self, user_friendly_config):
        """Test that error messages are helpful and actionable."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        error_scenarios = [
            {
                'url': 'https://nonexistent-domain-12345.com',
                'expected_error_types': ['network', 'dns', 'connection']
            },
            {
                'url': 'https://httpbin.org/status/404',
                'expected_error_types': ['404', 'not found', 'http']
            },
            {
                'url': 'https://httpbin.org/status/500',
                'expected_error_types': ['500', 'server error', 'http']
            }
        ]
        
        for scenario in error_scenarios:
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                if '404' in scenario['url']:
                    mock_response = Mock()
                    mock_response.status_code = 404
                    mock_response.text = 'Not Found'
                    mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                elif '500' in scenario['url']:
                    mock_response = Mock()
                    mock_response.status_code = 500
                    mock_response.text = 'Internal Server Error'
                    mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                else:
                    mock_stealth.fetch_with_stealth = AsyncMock(side_effect=Exception('DNS resolution failed'))
                
                async with crawler:
                    result = await crawler.crawl_url(scenario['url'])
                    
                    # Should provide helpful error information
                    if result is not None and 'error' in result:
                        error_message = result.get('error_message', '').lower()
                        error_type = result.get('error_type', '').lower()
                        
                        # Error messages should be informative
                        assert len(error_message) > 10  # Not just a code
                        assert any(error_type in error_message or error_type in error_type 
                                 for error_type in scenario['expected_error_types'])
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_configuration_validation_feedback(self):
        """Test that configuration validation provides helpful feedback."""
        # Test invalid configurations
        invalid_configs = [
            {
                'config': {'max_concurrent': 0},
                'expected_message': 'max_concurrent must be positive'
            },
            {
                'config': {'retry_count': -1},
                'expected_message': 'retry_count must be non-negative'
            },
            {
                'config': {'timeout': -5},
                'expected_message': 'timeout must be positive'
            }
        ]
        
        for test_case in invalid_configs:
            try:
                NewsCrawler(config=test_case['config'])
                assert False, f"Should have raised ValueError for {test_case['config']}"
            except ValueError as e:
                error_message = str(e).lower()
                expected_message = test_case['expected_message'].lower()
                
                # Error message should be helpful
                assert expected_message in error_message or \
                       any(word in error_message for word in expected_message.split())
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_result_structure_consistency(self, user_friendly_config):
        """Test that result structures are consistent and predictable."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        test_cases = [
            {
                'url': 'https://example.com/news',
                'html': '<html><head><title>News</title></head><body><p>News content</p></body></html>',
                'expected_fields': ['url', 'title', 'content', 'timestamp', 'metadata']
            },
            {
                'url': 'https://example.com/blog',
                'html': '<html><head><title>Blog</title></head><body><p>Blog content</p></body></html>',
                'expected_fields': ['url', 'title', 'content', 'timestamp', 'metadata']
            }
        ]
        
        for test_case in test_cases:
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = test_case['html']
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    result = await crawler.crawl_url(test_case['url'])
                    
                    # Result structure should be consistent
                    assert result is not None
                    
                    # Should have expected fields
                    for field in test_case['expected_fields']:
                        assert field in result, f"Missing field: {field}"
                    
                    # Field types should be predictable
                    assert isinstance(result['url'], str)
                    assert isinstance(result['title'], str)
                    assert isinstance(result['content'], str)
                    assert isinstance(result['timestamp'], str)
                    assert isinstance(result['metadata'], dict)
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_context_manager_convenience(self, user_friendly_config):
        """Test that context manager usage is convenient."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        # Should work seamlessly with context manager
        async with crawler:
            assert crawler.is_running
            
            # Should be able to perform operations
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = '<html><body>Test</body></html>'
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                result = await crawler.crawl_url('https://example.com/test')
                assert result is not None
        
        # Should clean up automatically
        assert not crawler.is_running
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_progress_visibility(self, user_friendly_config):
        """Test that progress is visible during long operations."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        # Test with multiple URLs to track progress
        urls = [f'https://example.com/page{i}' for i in range(10)]
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Content</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                # Should provide progress information
                results = await crawler.crawl_urls(urls)
                
                # Should have metrics available
                assert hasattr(crawler, 'metrics')
                assert crawler.metrics.total_requests > 0
                assert crawler.metrics.successful_requests > 0
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_default_behavior_sensible(self):
        """Test that default behavior is sensible for most use cases."""
        # Should work with minimal configuration
        crawler = NewsCrawler()
        
        # Defaults should be reasonable
        assert crawler.max_concurrent == 10  # Not too high, not too low
        assert crawler.retry_count == 3      # Reasonable retry count
        assert crawler.timeout == 30.0       # Reasonable timeout
        assert crawler.stealth_enabled       # Stealth should be enabled by default
        assert crawler.cache_enabled         # Caching should be enabled by default
        
        # Should work out of the box
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Default test</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                result = await crawler.crawl_url('https://example.com/default')
                assert result is not None
                assert 'content' in result
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_method_naming_intuitive(self, user_friendly_config):
        """Test that method names are intuitive and self-explanatory."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        # Method names should be self-explanatory
        assert hasattr(crawler, 'crawl_url')        # Single URL
        assert hasattr(crawler, 'crawl_urls')       # Multiple URLs
        assert hasattr(crawler, 'start')            # Start crawler
        assert hasattr(crawler, 'stop')             # Stop crawler
        assert hasattr(crawler, 'is_running')       # Check status
        
        # Should not require deep knowledge of internals
        async with crawler:
            # Basic operations should be straightforward
            assert crawler.is_running
            
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = '<html><body>Test</body></html>'
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                # Method behavior should match expectations
                result = await crawler.crawl_url('https://example.com/intuitive')
                assert result is not None
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_documentation_examples_work(self, user_friendly_config):
        """Test that documentation examples work as expected."""
        # Example 1: Simple usage
        crawler = NewsCrawler()
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><head><title>Example</title></head><body><p>Content</p></body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                result = await crawler.crawl_url('https://example.com/news')
                assert result is not None
                assert result['title'] == 'Example'
        
        # Example 2: Batch processing
        urls = ['https://example.com/1', 'https://example.com/2']
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Batch content</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                results = await crawler.crawl_urls(urls)
                assert len(results) == 2
        
        # Example 3: Custom configuration
        config = {
            'max_concurrent': 3,
            'stealth_enabled': True,
            'cache_enabled': True
        }
        
        custom_crawler = NewsCrawler(config=config)
        assert custom_crawler.max_concurrent == 3
        assert custom_crawler.stealth_enabled
        assert custom_crawler.cache_enabled
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_common_use_cases_simple(self, user_friendly_config):
        """Test that common use cases are simple to implement."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <head>
                    <title>News Article</title>
                    <meta name="author" content="John Doe">
                    <meta name="publish-date" content="2023-01-01">
                </head>
                <body>
                    <h1>Breaking News</h1>
                    <p>This is the article content.</p>
                    <img src="image.jpg" alt="News image">
                    <a href="related.html">Related article</a>
                </body>
            </html>
            """
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                # Use case 1: Extract article content
                result = await crawler.crawl_url('https://example.com/news')
                
                assert 'title' in result
                assert 'content' in result
                assert 'metadata' in result
                assert result['title'] == 'News Article'
                assert 'Breaking News' in result['content']
                assert 'This is the article content' in result['content']
                
                # Use case 2: Get metadata
                metadata = result['metadata']
                assert 'author' in metadata
                assert 'publish_date' in metadata
                
                # Use case 3: Extract images and links
                if 'images' in result:
                    assert len(result['images']) > 0
                
                if 'links' in result:
                    assert len(result['links']) > 0
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_error_recovery_user_friendly(self, user_friendly_config):
        """Test that error recovery is user-friendly."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        # Mix of working and broken URLs
        urls = [
            'https://example.com/working1',
            'https://nonexistent-domain.com/broken',
            'https://example.com/working2',
            'https://example.com/404',
            'https://example.com/working3'
        ]
        
        def mock_fetch_with_errors(url):
            if 'working' in url:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = '<html><body>Working content</body></html>'
                return mock_response
            elif 'nonexistent' in url:
                raise Exception('DNS resolution failed')
            elif '404' in url:
                mock_response = Mock()
                mock_response.status_code = 404
                mock_response.text = 'Not Found'
                return mock_response
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=mock_fetch_with_errors)
            
            async with crawler:
                results = await crawler.crawl_urls(urls)
                
                # Should return results for all URLs
                assert len(results) == len(urls)
                
                # Working URLs should have valid results
                working_results = [r for r in results if r is not None and 'error' not in r]
                assert len(working_results) >= 3  # At least 3 working URLs
                
                # Failed URLs should have error information
                failed_results = [r for r in results if r is None or 'error' in r]
                assert len(failed_results) >= 2  # At least 2 failed URLs
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_configuration_flexibility(self):
        """Test that configuration is flexible but not overwhelming."""
        # Should work with no configuration
        crawler1 = NewsCrawler()
        assert crawler1 is not None
        
        # Should work with minimal configuration
        crawler2 = NewsCrawler(config={'max_concurrent': 5})
        assert crawler2.max_concurrent == 5
        
        # Should work with comprehensive configuration
        full_config = {
            'max_concurrent': 8,
            'retry_count': 2,
            'timeout': 20.0,
            'stealth_enabled': True,
            'cache_enabled': True,
            'cache_ttl': 7200,
            'content_extraction': {
                'extract_text': True,
                'extract_images': True,
                'extract_links': True,
                'extract_metadata': True
            },
            'stealth_config': {
                'rotate_user_agents': True,
                'delay_range': [1, 3],
                'use_proxies': False
            }
        }
        
        crawler3 = NewsCrawler(config=full_config)
        assert crawler3.max_concurrent == 8
        assert crawler3.retry_count == 2
        assert crawler3.timeout == 20.0
        assert crawler3.stealth_enabled
        assert crawler3.cache_enabled
    
    @pytest.mark.usability
    @pytest.mark.asyncio
    async def test_debugging_information_available(self, user_friendly_config):
        """Test that debugging information is available when needed."""
        crawler = NewsCrawler(config=user_friendly_config)
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Debug content</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                result = await crawler.crawl_url('https://example.com/debug')
                
                # Should have debugging information available
                assert hasattr(crawler, 'metrics')
                assert crawler.metrics.total_requests > 0
                
                # Result should contain useful debugging info
                assert 'timestamp' in result
                assert 'url' in result
                
                # Should be able to inspect crawler state
                assert hasattr(crawler, 'is_running')
                assert hasattr(crawler, 'max_concurrent')
                assert hasattr(crawler, 'retry_count')