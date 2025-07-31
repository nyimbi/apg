"""
Integration tests for news crawler components.

This module contains comprehensive integration tests that validate
the interaction between different news crawler components.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from typing import Dict, Any, List
import tempfile
import os

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.crawlers.news_crawler.bypass.bypass_manager import BypassManager
from packages.crawlers.news_crawler.parsers.content_parser import ContentParser
from packages.crawlers.news_crawler.stealth.unified_stealth_orchestrator import UnifiedStealthOrchestrator
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


class TestNewsCrawlerIntegration:
    """Integration tests for news crawler system."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def mock_db_manager(self):
        """Mock database manager for testing."""
        mock_db = Mock(spec=PgSQLManager)
        mock_db.store_article = AsyncMock(return_value={"id": "test-id"})
        mock_db.get_article = AsyncMock(return_value=None)
        mock_db.update_article = AsyncMock()
        mock_db.is_connected = Mock(return_value=True)
        return mock_db
    
    @pytest.fixture
    def mock_cache_manager(self, temp_cache_dir):
        """Mock cache manager for testing."""
        mock_cache = Mock(spec=CacheManager)
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.set = AsyncMock()
        mock_cache.delete = AsyncMock()
        mock_cache.clear = AsyncMock()
        mock_cache.cache_dir = temp_cache_dir
        return mock_cache
    
    @pytest.fixture
    def crawler_config(self):
        """Configuration for news crawler integration tests."""
        return {
            'max_concurrent': 3,
            'retry_count': 2,
            'timeout': 10.0,
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
                'delay_range': [1, 3]
            }
        }
    
    @pytest.fixture
    def news_crawler(self, crawler_config, mock_db_manager, mock_cache_manager):
        """Create news crawler instance for integration testing."""
        crawler = NewsCrawler(config=crawler_config)
        # Mock the internal components
        crawler.db_manager = mock_db_manager
        crawler.cache_manager = mock_cache_manager
        return crawler
    
    @pytest.mark.asyncio
    async def test_full_crawl_workflow(self, news_crawler, mock_db_manager):
        """Test complete crawl workflow from URL to database storage."""
        test_url = "https://httpbin.org/html"
        
        # Mock stealth orchestrator
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = """
            <html>
                <head>
                    <title>Test Article</title>
                    <meta name="description" content="Test description">
                </head>
                <body>
                    <h1>Test Headline</h1>
                    <p>Test content paragraph 1.</p>
                    <p>Test content paragraph 2.</p>
                </body>
            </html>
            """
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            # Execute crawl
            result = await news_crawler.crawl_url(test_url)
            
            # Verify result structure
            assert result is not None
            assert 'url' in result
            assert 'title' in result
            assert 'content' in result
            assert 'metadata' in result
            assert result['url'] == test_url
            assert result['title'] == "Test Article"
            assert "Test content paragraph" in result['content']
            
            # Verify database storage was called
            mock_db_manager.store_article.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stealth_bypass_integration(self, news_crawler):
        """Test integration between stealth and bypass components."""
        test_url = "https://httpbin.org/status/403"
        
        # Mock initial 403 response
        mock_403_response = Mock()
        mock_403_response.status_code = 403
        mock_403_response.text = "Forbidden"
        
        # Mock successful bypass response
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.text = "<html><body>Success</body></html>"
        
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            with patch.object(news_crawler, 'bypass_manager') as mock_bypass:
                # First call returns 403, second call returns 200
                mock_stealth.fetch_with_stealth.side_effect = [
                    mock_403_response,
                    mock_success_response
                ]
                mock_bypass.handle_403_error = AsyncMock(return_value=True)
                
                result = await news_crawler.crawl_url(test_url)
                
                # Verify bypass was triggered
                mock_bypass.handle_403_error.assert_called_once()
                
                # Verify second attempt succeeded
                assert result is not None
                assert mock_stealth.fetch_with_stealth.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, news_crawler, mock_cache_manager):
        """Test cache integration in crawl workflow."""
        test_url = "https://httpbin.org/json"
        
        # Mock cached response
        cached_result = {
            'url': test_url,
            'title': 'Cached Article',
            'content': 'Cached content',
            'metadata': {'cached': True}
        }
        
        mock_cache_manager.get.return_value = cached_result
        
        result = await news_crawler.crawl_url(test_url)
        
        # Verify cached result was returned
        assert result == cached_result
        mock_cache_manager.get.assert_called_once()
        
        # Verify no HTTP request was made
        assert not hasattr(news_crawler, 'stealth_orchestrator') or \
               not news_crawler.stealth_orchestrator.fetch_with_stealth.called
    
    @pytest.mark.asyncio
    async def test_content_parser_integration(self, news_crawler):
        """Test content parser integration with various content types."""
        test_cases = [
            {
                'url': 'https://example.com/article1',
                'html': '''
                <html>
                    <head><title>News Article</title></head>
                    <body>
                        <article>
                            <h1>Main Headline</h1>
                            <p class="lead">Lead paragraph</p>
                            <div class="content">
                                <p>Content paragraph 1</p>
                                <p>Content paragraph 2</p>
                            </div>
                        </article>
                    </body>
                </html>
                ''',
                'expected_title': 'News Article',
                'expected_content_contains': ['Lead paragraph', 'Content paragraph']
            },
            {
                'url': 'https://example.com/article2',
                'html': '''
                <html>
                    <head><title>Blog Post</title></head>
                    <body>
                        <div class="post">
                            <h2>Blog Title</h2>
                            <p>Blog content goes here.</p>
                        </div>
                    </body>
                </html>
                ''',
                'expected_title': 'Blog Post',
                'expected_content_contains': ['Blog content goes here']
            }
        ]
        
        for test_case in test_cases:
            with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = test_case['html']
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                result = await news_crawler.crawl_url(test_case['url'])
                
                assert result['title'] == test_case['expected_title']
                for expected_content in test_case['expected_content_contains']:
                    assert expected_content in result['content']
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, news_crawler, mock_db_manager):
        """Test error handling across integrated components."""
        test_url = "https://nonexistent-domain-12345.com"
        
        # Mock network error
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=Exception("Network error"))
            
            result = await news_crawler.crawl_url(test_url)
            
            # Should handle error gracefully
            assert result is None or 'error' in result
            
            # Should not attempt database storage on error
            mock_db_manager.store_article.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_crawl_integration(self, news_crawler):
        """Test concurrent crawling with integrated components."""
        test_urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/2",
            "https://httpbin.org/delay/1"
        ]
        
        # Mock responses
        mock_responses = []
        for i, url in enumerate(test_urls):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body>Response {i}</body></html>"
            mock_responses.append(mock_response)
        
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=mock_responses)
            
            start_time = datetime.now()
            results = await news_crawler.crawl_urls(test_urls)
            end_time = datetime.now()
            
            # Should complete faster than sequential processing
            duration = (end_time - start_time).total_seconds()
            assert duration < 4  # Should be faster than sum of delays
            
            # Should return results for all URLs
            assert len(results) == len(test_urls)
            assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_database_integration_with_metadata(self, news_crawler, mock_db_manager):
        """Test database integration with extracted metadata."""
        test_url = "https://example.com/article"
        
        # Mock response with rich metadata
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
            <head>
                <title>Article Title</title>
                <meta name="description" content="Article description">
                <meta name="author" content="John Doe">
                <meta name="publish-date" content="2023-01-01">
                <meta property="og:image" content="https://example.com/image.jpg">
            </head>
            <body>
                <article>
                    <h1>Article Title</h1>
                    <p>Article content</p>
                </article>
            </body>
        </html>
        '''
        
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            result = await news_crawler.crawl_url(test_url)
            
            # Verify metadata extraction
            assert 'metadata' in result
            metadata = result['metadata']
            assert metadata.get('description') == 'Article description'
            assert metadata.get('author') == 'John Doe'
            assert metadata.get('publish_date') == '2023-01-01'
            assert metadata.get('og_image') == 'https://example.com/image.jpg'
            
            # Verify database storage includes metadata
            mock_db_manager.store_article.assert_called_once()
            stored_data = mock_db_manager.store_article.call_args[0][0]
            assert 'metadata' in stored_data
    
    @pytest.mark.asyncio
    async def test_retry_mechanism_integration(self, news_crawler):
        """Test retry mechanism across integrated components."""
        test_url = "https://httpbin.org/status/500"
        
        # Mock responses: first two fail, third succeeds
        mock_responses = [
            Mock(status_code=500, text="Server Error"),
            Mock(status_code=503, text="Service Unavailable"),
            Mock(status_code=200, text="<html><body>Success</body></html>")
        ]
        
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=mock_responses)
            
            result = await news_crawler.crawl_url(test_url)
            
            # Should eventually succeed after retries
            assert result is not None
            assert 'Success' in result.get('content', '')
            
            # Should have made 3 attempts
            assert mock_stealth.fetch_with_stealth.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, news_crawler):
        """Test rate limiting integration across components."""
        test_urls = [f"https://httpbin.org/json?id={i}" for i in range(10)]
        
        # Mock responses
        mock_responses = []
        for i in range(10):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = f"<html><body>Response {i}</body></html>"
            mock_responses.append(mock_response)
        
        with patch.object(news_crawler, 'stealth_orchestrator') as mock_stealth:
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=mock_responses)
            
            # Set aggressive rate limiting
            news_crawler.config['rate_limit'] = 2
            news_crawler.config['rate_window'] = 1
            
            start_time = datetime.now()
            results = await news_crawler.crawl_urls(test_urls)
            end_time = datetime.now()
            
            # Should take longer due to rate limiting
            duration = (end_time - start_time).total_seconds()
            assert duration > 4  # Should be rate limited
            
            # Should still process all URLs
            assert len(results) == len(test_urls)
    
    @pytest.mark.asyncio
    async def test_configuration_integration(self, mock_db_manager, mock_cache_manager):
        """Test configuration integration across components."""
        custom_config = {
            'max_concurrent': 1,
            'retry_count': 1,
            'timeout': 5.0,
            'stealth_enabled': False,
            'bypass_enabled': False,
            'cache_enabled': False,
            'content_extraction': {
                'extract_text': True,
                'extract_images': False,
                'extract_links': False,
                'extract_metadata': False
            }
        }
        
        crawler = NewsCrawler(config=custom_config)
        # Mock the internal components
        crawler.db_manager = mock_db_manager
        crawler.cache_manager = mock_cache_manager
        
        # Verify configuration propagation through config dict
        assert crawler.config['max_concurrent'] == 1
        assert crawler.config['max_retries'] == 1
        assert crawler.config['timeout'] == 5.0
        assert not crawler.config.get('stealth_enabled', True)
        assert not crawler.config.get('bypass_enabled', True)
        assert not crawler.config.get('cache_enabled', True)
        
        # Test crawl with limited extraction
        test_url = "https://httpbin.org/html"
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '''
        <html>
            <head><title>Test</title></head>
            <body>
                <p>Content</p>
                <img src="image.jpg" alt="Test">
                <a href="link.html">Link</a>
            </body>
        </html>
        '''
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value.status = 200
            mock_get.return_value.__aenter__.return_value.text = AsyncMock(return_value=mock_response.text)
            
            result = await crawler.crawl_url(test_url)
            
            # Should extract only text based on configuration
            assert 'content' in result
            assert 'images' not in result or len(result['images']) == 0
            assert 'links' not in result or len(result['links']) == 0
            assert 'metadata' not in result or len(result['metadata']) == 0