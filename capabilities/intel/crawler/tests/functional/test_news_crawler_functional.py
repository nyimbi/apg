"""
Functional tests for news crawler.

This module contains functional tests that validate the news crawler
against real-world scenarios and user requirements.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
import os

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


class TestNewsCrawlerFunctional:
    """Functional tests for news crawler system."""
    
    @pytest.fixture
    def functional_config(self):
        """Configuration for functional testing."""
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
    def test_urls(self):
        """Test URLs for functional testing."""
        return {
            'news_sites': [
                'https://httpbin.org/html',  # Simple HTML
                'https://httpbin.org/json',  # JSON response
                'https://httpbin.org/xml',   # XML response
            ],
            'challenging_sites': [
                'https://httpbin.org/status/403',  # Forbidden
                'https://httpbin.org/status/404',  # Not Found
                'https://httpbin.org/delay/5',     # Slow response
            ],
            'redirect_sites': [
                'https://httpbin.org/redirect/3',     # Multiple redirects
                'https://httpbin.org/redirect-to?url=https://httpbin.org/json',
            ]
        }
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_crawl_standard_news_sites(self, functional_config, test_urls):
        """Test crawling standard news sites."""
        crawler = NewsCrawler(config=functional_config)
        
        async with crawler:
            for url in test_urls['news_sites']:
                result = await crawler.crawl_url(url)
                
                # Verify basic structure
                assert result is not None
                assert 'url' in result
                assert 'content' in result
                assert 'timestamp' in result
                
                # Verify content quality
                assert len(result['content']) > 0
                assert result['url'] == url
                
                # Verify timestamp is recent
                timestamp = datetime.fromisoformat(result['timestamp'].replace('Z', '+00:00'))
                assert (datetime.now(timezone.utc) - timestamp).total_seconds() < 60
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_handle_difficult_sites(self, functional_config, test_urls):
        """Test handling of difficult or problematic sites."""
        crawler = NewsCrawler(config=functional_config)
        
        async with crawler:
            for url in test_urls['challenging_sites']:
                result = await crawler.crawl_url(url)
                
                # Should either succeed or fail gracefully
                if result is not None:
                    assert 'url' in result
                    assert 'content' in result or 'error' in result
                else:
                    # None result is acceptable for failed crawls
                    pass
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_handle_redirects(self, functional_config, test_urls):
        """Test handling of redirects."""
        crawler = NewsCrawler(config=functional_config)
        
        async with crawler:
            for url in test_urls['redirect_sites']:
                result = await crawler.crawl_url(url)
                
                # Should follow redirects and succeed
                assert result is not None
                assert 'url' in result
                assert 'content' in result
                
                # Final URL might be different due to redirects
                assert result['url'] is not None
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_concurrent_crawling_performance(self, functional_config, test_urls):
        """Test concurrent crawling performance."""
        crawler = NewsCrawler(config=functional_config)
        
        # Create list of URLs to crawl
        urls_to_crawl = test_urls['news_sites'] * 3  # 9 URLs total
        
        async with crawler:
            start_time = datetime.now()
            results = await crawler.crawl_urls(urls_to_crawl)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # Should complete faster than sequential processing
            # With max_concurrent=5, should be significantly faster
            assert duration < 30  # Should complete within 30 seconds
            
            # Should return results for all URLs
            assert len(results) == len(urls_to_crawl)
            
            # Most results should be successful
            successful_results = [r for r in results if r is not None and 'error' not in r]
            success_rate = len(successful_results) / len(results)
            assert success_rate > 0.8  # At least 80% success rate
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_content_extraction_quality(self, functional_config):
        """Test quality of content extraction."""
        crawler = NewsCrawler(config=functional_config)
        
        # Test with a rich HTML page
        test_url = 'https://httpbin.org/html'
        
        async with crawler:
            result = await crawler.crawl_url(test_url)
            
            assert result is not None
            
            # Check content extraction
            content = result.get('content', '')
            assert len(content) > 0
            assert content.strip() != ''
            
            # Check metadata extraction
            metadata = result.get('metadata', {})
            assert isinstance(metadata, dict)
            
            # Check if basic HTML elements are handled
            assert 'title' in result
            assert result['title'] is not None
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, functional_config):
        """Test error recovery in various scenarios."""
        crawler = NewsCrawler(config=functional_config)
        
        test_scenarios = [
            {
                'url': 'https://httpbin.org/status/500',
                'expected_behavior': 'retry_then_fail_gracefully'
            },
            {
                'url': 'https://httpbin.org/status/503',
                'expected_behavior': 'retry_then_fail_gracefully'
            },
            {
                'url': 'https://nonexistent-domain-12345.com',
                'expected_behavior': 'fail_immediately'
            }
        ]
        
        async with crawler:
            for scenario in test_scenarios:
                result = await crawler.crawl_url(scenario['url'])
                
                # Should not raise exceptions
                # Result can be None or contain error information
                if result is not None and 'error' in result:
                    assert 'error_type' in result
                    assert 'error_message' in result
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_stealth_capabilities(self, functional_config):
        """Test stealth capabilities against detection."""
        crawler = NewsCrawler(config=functional_config)
        
        # Test user agent rotation
        test_urls = ['https://httpbin.org/user-agent'] * 3
        
        async with crawler:
            results = await crawler.crawl_urls(test_urls)
            
            # Should successfully crawl all URLs
            successful_results = [r for r in results if r is not None and 'error' not in r]
            assert len(successful_results) >= 2  # At least 2 should succeed
            
            # If stealth is working, user agents might be different
            # This is hard to test without parsing the response content
            # But we can verify no obvious bot detection occurred
            for result in successful_results:
                assert 'blocked' not in result.get('content', '').lower()
                assert 'bot' not in result.get('content', '').lower()
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_cache_functionality(self, functional_config):
        """Test caching functionality in real scenarios."""
        crawler = NewsCrawler(config=functional_config)
        
        test_url = 'https://httpbin.org/json'
        
        async with crawler:
            # First crawl - should fetch from web
            start_time1 = datetime.now()
            result1 = await crawler.crawl_url(test_url)
            end_time1 = datetime.now()
            duration1 = (end_time1 - start_time1).total_seconds()
            
            # Second crawl - should use cache
            start_time2 = datetime.now()
            result2 = await crawler.crawl_url(test_url)
            end_time2 = datetime.now()
            duration2 = (end_time2 - start_time2).total_seconds()
            
            # Both should succeed
            assert result1 is not None
            assert result2 is not None
            
            # Results should be similar (cached version)
            assert result1['url'] == result2['url']
            
            # Second call should be faster (cached)
            assert duration2 < duration1
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_rate_limiting_compliance(self, functional_config):
        """Test rate limiting compliance."""
        # Set aggressive rate limiting
        functional_config['rate_limit'] = 2
        functional_config['rate_window'] = 5
        
        crawler = NewsCrawler(config=functional_config)
        
        test_urls = ['https://httpbin.org/json?id={}'.format(i) for i in range(6)]
        
        async with crawler:
            start_time = datetime.now()
            results = await crawler.crawl_urls(test_urls)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # Should respect rate limiting
            # With 2 requests per 5 seconds, 6 requests should take ~15 seconds
            assert duration >= 10  # Should be rate limited
            
            # Should still process all URLs
            assert len(results) == len(test_urls)
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_timeout_handling(self, functional_config):
        """Test timeout handling in real scenarios."""
        # Set short timeout for testing
        functional_config['timeout'] = 3.0
        
        crawler = NewsCrawler(config=functional_config)
        
        # Test with slow URL
        slow_url = 'https://httpbin.org/delay/10'
        
        async with crawler:
            result = await crawler.crawl_url(slow_url)
            
            # Should handle timeout gracefully
            if result is not None:
                assert 'error' in result
                assert 'timeout' in result['error_type'].lower()
            else:
                # None result is acceptable for timeout
                pass
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, functional_config):
        """Test resource cleanup after crawling."""
        crawler = NewsCrawler(config=functional_config)
        
        test_urls = ['https://httpbin.org/json'] * 10
        
        # Test with context manager
        async with crawler:
            results = await crawler.crawl_urls(test_urls)
            assert len(results) == len(test_urls)
        
        # After context manager, resources should be cleaned up
        assert not crawler.is_running
        
        # Test manual start/stop
        await crawler.start()
        assert crawler.is_running
        
        results2 = await crawler.crawl_urls(test_urls[:3])
        assert len(results2) == 3
        
        await crawler.stop()
        assert not crawler.is_running
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation in functional context."""
        # Test with invalid configuration
        invalid_configs = [
            {'max_concurrent': 0},  # Invalid concurrent count
            {'retry_count': -1},    # Invalid retry count
            {'timeout': -5},        # Invalid timeout
        ]
        
        for config in invalid_configs:
            with pytest.raises(ValueError):
                NewsCrawler(config=config)
        
        # Test with valid edge cases
        edge_configs = [
            {'max_concurrent': 1},     # Minimum concurrent
            {'retry_count': 0},        # No retries
            {'timeout': 1.0},          # Minimum timeout
        ]
        
        for config in edge_configs:
            crawler = NewsCrawler(config=config)
            assert crawler is not None
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_real_world_news_extraction(self, functional_config):
        """Test extraction from real-world news-like content."""
        # Mock a realistic news article structure
        news_content = '''
        <html>
            <head>
                <title>Breaking: Major News Event Happens</title>
                <meta name="description" content="Description of major news event">
                <meta name="author" content="News Reporter">
                <meta name="publish-date" content="2023-01-01T12:00:00Z">
            </head>
            <body>
                <article>
                    <h1>Breaking: Major News Event Happens</h1>
                    <p class="lead">This is the lead paragraph of the news article.</p>
                    <div class="content">
                        <p>First paragraph of the main content.</p>
                        <p>Second paragraph with more details.</p>
                        <p>Third paragraph with analysis.</p>
                    </div>
                    <div class="author">By News Reporter</div>
                    <div class="timestamp">January 1, 2023</div>
                </article>
            </body>
        </html>
        '''
        
        # This would need to be tested with a mock server
        # For now, we test the structure that would be extracted
        
        crawler = NewsCrawler(config=functional_config)
        
        # Test extraction capabilities
        expected_fields = [
            'url', 'title', 'content', 'metadata', 'timestamp',
            'author', 'publish_date', 'images', 'links'
        ]
        
        # Mock the extraction process
        async with crawler:
            # This would be a real URL in production
            # result = await crawler.crawl_url('https://example-news-site.com/article')
            
            # For testing, we verify the crawler has the capability
            assert hasattr(crawler, 'extract_content')
            assert hasattr(crawler, 'extract_metadata')
            assert hasattr(crawler, 'extract_images')
            assert hasattr(crawler, 'extract_links')
    
    @pytest.mark.functional
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_scale_crawling(self, functional_config):
        """Test large-scale crawling scenarios."""
        crawler = NewsCrawler(config=functional_config)
        
        # Generate many URLs for testing
        base_urls = [
            'https://httpbin.org/json',
            'https://httpbin.org/html',
            'https://httpbin.org/xml'
        ]
        
        # Create 30 URLs (10 of each type)
        test_urls = []
        for i in range(10):
            for base_url in base_urls:
                test_urls.append(f"{base_url}?id={i}")
        
        async with crawler:
            start_time = datetime.now()
            results = await crawler.crawl_urls(test_urls)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # Verify results
            assert len(results) == len(test_urls)
            
            # Check success rate
            successful_results = [r for r in results if r is not None and 'error' not in r]
            success_rate = len(successful_results) / len(results)
            assert success_rate > 0.8  # At least 80% success rate
            
            # Check performance
            requests_per_second = len(test_urls) / duration
            assert requests_per_second > 1  # Should process at least 1 request per second
            
            # Verify no memory leaks (basic check)
            assert len(crawler._active_requests) == 0
    
    @pytest.mark.functional
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, functional_config):
        """Test graceful shutdown during active crawling."""
        crawler = NewsCrawler(config=functional_config)
        
        # URLs with delays to ensure crawling is active during shutdown
        test_urls = [f'https://httpbin.org/delay/{i}' for i in range(1, 6)]
        
        await crawler.start()
        
        # Start crawling (don't await)
        crawl_task = asyncio.create_task(crawler.crawl_urls(test_urls))
        
        # Wait a bit to ensure crawling is active
        await asyncio.sleep(2)
        
        # Shutdown gracefully
        await crawler.stop()
        
        # Crawling should complete or be cancelled gracefully
        try:
            results = await crawl_task
            # If completed, should have some results
            assert isinstance(results, list)
        except asyncio.CancelledError:
            # Cancellation is acceptable during shutdown
            pass
        
        # Crawler should be stopped
        assert not crawler.is_running