"""
Unit tests for simple crawler functionality.

This module contains basic unit tests for the SimpleCrawler class,
testing individual components work correctly in isolation.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone
from typing import Dict, Any, List

from packages.crawlers.base_crawler import (
    SimpleCrawler,
    RequestPriority,
    CrawlRequest,
    CrawlResponse
)


class TestSimpleCrawler:
    """Test suite for SimpleCrawler class."""
    
    @pytest.fixture
    def crawler(self) -> SimpleCrawler:
        """Create a SimpleCrawler instance for testing."""
        return SimpleCrawler(
            rate_limit=10.0,
            max_concurrent=5,
            timeout=30,
            max_retries=3
        )
    
    def test_crawler_initialization_with_params(self):
        """Test crawler initialization with parameters."""
        crawler = SimpleCrawler(
            rate_limit=5.0,
            max_concurrent=3,
            timeout=20,
            max_retries=2
        )
        
        assert crawler.rate_limit == 5.0
        assert crawler.max_concurrent == 3
        assert crawler.timeout == 20
        assert crawler.max_retries == 2
    
    def test_crawler_initialization_with_defaults(self):
        """Test crawler initialization with default values."""
        crawler = SimpleCrawler()
        
        assert crawler.rate_limit == 10.0
        assert crawler.max_concurrent == 10
        assert crawler.timeout == 30
        assert crawler.max_retries == 3
        assert crawler.cache_responses == False
    
    def test_crawler_headers_initialization(self):
        """Test that default headers are set correctly."""
        crawler = SimpleCrawler()
        
        assert 'User-Agent' in crawler.default_headers
        assert 'Accept' in crawler.default_headers
        assert 'Accept-Language' in crawler.default_headers
        assert crawler.default_headers['DNT'] == '1'
        assert crawler.default_headers['Connection'] == 'keep-alive'
    
    def test_crawler_custom_headers(self):
        """Test initialization with custom headers."""
        custom_headers = {
            'Custom-Header': 'test-value',
            'Authorization': 'Bearer token123'
        }
        
        crawler = SimpleCrawler(headers=custom_headers)
        
        # Should have both default and custom headers
        assert 'User-Agent' in crawler.default_headers
        assert 'Custom-Header' in crawler.default_headers
        assert 'Authorization' in crawler.default_headers
        assert crawler.default_headers['Custom-Header'] == 'test-value'
        assert crawler.default_headers['Authorization'] == 'Bearer token123'
    
    def test_crawler_custom_user_agent(self):
        """Test initialization with custom user agent."""
        custom_ua = "TestBot/1.0"
        crawler = SimpleCrawler(user_agent=custom_ua)
        
        assert crawler.default_headers['User-Agent'] == custom_ua
    
    def test_crawler_cache_configuration(self):
        """Test cache configuration."""
        # Without caching
        crawler1 = SimpleCrawler(cache_responses=False)
        assert crawler1.cache_responses == False
        
        # With caching
        crawler2 = SimpleCrawler(cache_responses=True)
        assert crawler2.cache_responses == True
        assert crawler2.cache_dir is not None
    
    def test_rate_limiter_initialization(self, crawler):
        """Test that rate limiter is properly initialized."""
        assert hasattr(crawler, 'rate_limiter')
        assert crawler.rate_limiter is not None
        # Rate limiter should have the configured rate
        assert crawler.rate_limiter.rate == 10.0
    
    def test_connection_pool_initialization(self, crawler):
        """Test that connection pool is properly initialized."""
        assert hasattr(crawler, 'connection_pool')
        assert crawler.connection_pool is not None
    
    @pytest.mark.asyncio
    async def test_process_response_method(self, crawler):
        """Test the process_response method."""
        # Create a mock request first
        mock_request = CrawlRequest(
            url="https://example.com",
            method="GET"
        )
        
        # Create a mock response
        mock_response = CrawlResponse(
            request=mock_request,
            status_code=200,
            content="Test content",
            headers={'Content-Type': 'text/html'}
        )
        
        # SimpleCrawler should return the response unchanged
        result = await crawler.process_response(mock_response)
        
        assert result == mock_response
        assert result.request.url == "https://example.com"
        assert result.status_code == 200
        assert result.content == "Test content"
    
    def test_request_priority_enum(self):
        """Test RequestPriority enum values."""
        assert RequestPriority.CRITICAL.value == 1
        assert RequestPriority.HIGH.value == 2
        assert RequestPriority.NORMAL.value == 3
        assert RequestPriority.LOW.value == 4
        assert RequestPriority.BACKGROUND.value == 5
    
    def test_crawl_request_creation(self):
        """Test CrawlRequest creation."""
        request = CrawlRequest(
            url="https://example.com",
            method="GET",
            headers={"Custom": "header"},
            priority=RequestPriority.HIGH
        )
        
        assert request.url == "https://example.com"
        assert request.method == "GET"
        assert request.headers["Custom"] == "header"
        assert request.priority == RequestPriority.HIGH
    
    def test_crawl_response_creation(self):
        """Test CrawlResponse creation."""
        # Create a request first
        request = CrawlRequest(
            url="https://example.com",
            method="GET"
        )
        
        response = CrawlResponse(
            request=request,
            status_code=200,
            content="Test content",
            headers={'Content-Type': 'text/html'},
            encoding='utf-8'
        )
        
        assert response.request.url == "https://example.com"
        assert response.status_code == 200
        assert response.content == "Test content"
        assert response.headers['Content-Type'] == 'text/html'
        assert response.encoding == 'utf-8'
    
    @pytest.mark.asyncio
    async def test_context_manager_support(self):
        """Test that SimpleCrawler can be used as async context manager."""
        async with SimpleCrawler() as crawler:
            assert crawler is not None
            # Context manager should set up the crawler properly
            assert hasattr(crawler, 'rate_limiter')
            assert hasattr(crawler, 'connection_pool')
    
    def test_configuration_flexibility(self):
        """Test various configuration combinations."""
        # Minimal configuration
        crawler1 = SimpleCrawler()
        assert crawler1.max_concurrent == 10
        
        # Custom rate limiting
        crawler2 = SimpleCrawler(rate_limit=5.0, max_concurrent=3)
        assert crawler2.rate_limit == 5.0
        assert crawler2.max_concurrent == 3
        
        # Custom timeout and retries
        crawler3 = SimpleCrawler(timeout=60, max_retries=5)
        assert crawler3.timeout == 60
        assert crawler3.max_retries == 5
    
    def test_crawler_string_representation(self, crawler):
        """Test string representation of crawler."""
        str_repr = str(crawler)
        assert "SimpleCrawler" in str_repr or "Crawler" in str_repr
    
    def test_boundary_values(self):
        """Test crawler with boundary values."""
        # Minimum values
        crawler1 = SimpleCrawler(
            rate_limit=0.1,
            max_concurrent=1,
            timeout=1,
            max_retries=0
        )
        assert crawler1.rate_limit == 0.1
        assert crawler1.max_concurrent == 1
        assert crawler1.timeout == 1
        assert crawler1.max_retries == 0
        
        # Large values
        crawler2 = SimpleCrawler(
            rate_limit=1000.0,
            max_concurrent=100,
            timeout=3600,
            max_retries=10
        )
        assert crawler2.rate_limit == 1000.0
        assert crawler2.max_concurrent == 100
        assert crawler2.timeout == 3600
        assert crawler2.max_retries == 10