"""
Performance tests for news crawler system.

This module contains performance tests that validate the news crawler
behavior under various load conditions and stress scenarios.
"""

import pytest
import asyncio
import time
import psutil
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
import statistics

from packages.crawlers.news_crawler.core.enhanced_news_crawler import NewsCrawler
from packages.crawlers.base_crawler import BaseCrawler
from packages.database.postgresql_manager import PgSQLManager
from packages.utils.caching.manager import CacheManager


class TestNewsCrawlerPerformance:
    """Performance tests for news crawler system."""
    
    @pytest.fixture
    def performance_config(self):
        """Configuration optimized for performance testing."""
        return {
            'max_concurrent': 10,
            'retry_count': 2,
            'timeout': 15.0,
            'stealth_enabled': True,
            'bypass_enabled': True,
            'cache_enabled': True,
            'cache_ttl': 3600,
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
    def load_test_urls(self):
        """Generate URLs for load testing."""
        base_urls = [
            'https://httpbin.org/json',
            'https://httpbin.org/html',
            'https://httpbin.org/xml',
            'https://httpbin.org/user-agent',
            'https://httpbin.org/headers'
        ]
        
        # Generate 100 unique URLs
        urls = []
        for i in range(20):
            for base_url in base_urls:
                urls.append(f"{base_url}?id={i}")
        
        return urls
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_processing_performance(self, performance_config, load_test_urls):
        """Test concurrent processing performance."""
        crawler = NewsCrawler(config=performance_config)
        
        # Test with different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        results = {}
        
        for concurrency in concurrency_levels:
            crawler.config['max_concurrent'] = concurrency
            
            # Use subset of URLs for faster testing
            test_urls = load_test_urls[:20]
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.text = '{"test": "data"}'
                mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
                
                async with crawler:
                    crawl_results = await crawler.crawl_urls(test_urls)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_usage = end_memory - start_memory
            
            results[concurrency] = {
                'duration': duration,
                'memory_usage': memory_usage,
                'throughput': len(test_urls) / duration,
                'success_count': len([r for r in crawl_results if r is not None])
            }
        
        # Verify performance improves with concurrency
        assert results[10]['throughput'] > results[1]['throughput']
        assert results[10]['duration'] < results[1]['duration']
        
        # Verify memory usage is reasonable
        for concurrency, result in results.items():
            assert result['memory_usage'] < 100  # Less than 100MB increase
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, performance_config):
        """Test memory usage stability during extended operations."""
        crawler = NewsCrawler(config=performance_config)
        
        memory_samples = []
        test_urls = [f'https://httpbin.org/json?batch={i}' for i in range(50)]
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                for i in range(5):  # 5 batches
                    batch_urls = test_urls[i*10:(i+1)*10]
                    
                    # Measure memory before batch
                    memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    await crawler.crawl_urls(batch_urls)
                    
                    # Measure memory after batch
                    memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(memory_after - memory_before)
                    
                    # Small delay to allow garbage collection
                    await asyncio.sleep(0.1)
        
        # Check for memory leaks
        memory_trend = statistics.mean(memory_samples[-3:]) - statistics.mean(memory_samples[:3])
        assert memory_trend < 10  # Less than 10MB increase over time
        
        # Check memory usage is consistent
        memory_std = statistics.stdev(memory_samples)
        assert memory_std < 5  # Less than 5MB standard deviation
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_throughput_under_load(self, performance_config, load_test_urls):
        """Test throughput under high load conditions."""
        crawler = NewsCrawler(config=performance_config)
        
        # Test with large number of URLs
        test_urls = load_test_urls  # 100 URLs
        
        start_time = time.time()
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><body>Test content</body></html>'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                results = await crawler.crawl_urls(test_urls)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate performance metrics
        throughput = len(test_urls) / duration
        success_rate = len([r for r in results if r is not None]) / len(test_urls)
        
        # Performance assertions
        assert throughput > 5  # At least 5 requests per second
        assert success_rate > 0.9  # At least 90% success rate
        assert duration < 30  # Should complete within 30 seconds
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_rate_limiting_performance(self, performance_config):
        """Test performance with rate limiting enabled."""
        # Set strict rate limiting
        performance_config['rate_limit'] = 5
        performance_config['rate_window'] = 1
        
        crawler = NewsCrawler(config=performance_config)
        test_urls = [f'https://httpbin.org/json?rate={i}' for i in range(20)]
        
        start_time = time.time()
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                results = await crawler.crawl_urls(test_urls)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be rate limited but still complete
        expected_minimum_duration = (len(test_urls) / 5) * 0.8  # 80% of theoretical minimum
        assert duration >= expected_minimum_duration
        assert len(results) == len(test_urls)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, performance_config):
        """Test cache performance impact."""
        test_urls = [f'https://httpbin.org/json?cache={i}' for i in range(10)]
        
        # Test without cache
        performance_config['cache_enabled'] = False
        crawler_no_cache = NewsCrawler(config=performance_config)
        
        start_time = time.time()
        
        with patch.object(crawler_no_cache, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler_no_cache:
                # First run
                await crawler_no_cache.crawl_urls(test_urls)
                # Second run
                await crawler_no_cache.crawl_urls(test_urls)
        
        no_cache_duration = time.time() - start_time
        
        # Test with cache
        performance_config['cache_enabled'] = True
        crawler_with_cache = NewsCrawler(config=performance_config)
        
        start_time = time.time()
        
        with patch.object(crawler_with_cache, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler_with_cache:
                # First run - populates cache
                await crawler_with_cache.crawl_urls(test_urls)
                # Second run - should use cache
                await crawler_with_cache.crawl_urls(test_urls)
        
        with_cache_duration = time.time() - start_time
        
        # Cache should improve performance on second run
        assert with_cache_duration < no_cache_duration
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_error_handling_performance(self, performance_config):
        """Test performance impact of error handling."""
        # Mix of successful and failing URLs
        test_urls = [
            'https://httpbin.org/json',
            'https://httpbin.org/status/404',
            'https://httpbin.org/status/500',
            'https://httpbin.org/json',
            'https://httpbin.org/status/503',
            'https://httpbin.org/json',
            'https://httpbin.org/delay/2',
            'https://httpbin.org/json'
        ]
        
        crawler = NewsCrawler(config=performance_config)
        
        start_time = time.time()
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            def mock_fetch(url):
                if 'status/404' in url:
                    mock_response = Mock()
                    mock_response.status_code = 404
                    mock_response.text = 'Not Found'
                    return mock_response
                elif 'status/500' in url:
                    mock_response = Mock()
                    mock_response.status_code = 500
                    mock_response.text = 'Server Error'
                    return mock_response
                elif 'status/503' in url:
                    mock_response = Mock()
                    mock_response.status_code = 503
                    mock_response.text = 'Service Unavailable'
                    return mock_response
                else:
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.text = '{"test": "data"}'
                    return mock_response
            
            mock_stealth.fetch_with_stealth = AsyncMock(side_effect=mock_fetch)
            
            async with crawler:
                results = await crawler.crawl_urls(test_urls)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle errors gracefully without significant performance impact
        assert duration < 15  # Should complete within 15 seconds
        assert len(results) == len(test_urls)
        
        # Check that we got some successful results
        successful_results = [r for r in results if r is not None and 'error' not in str(r)]
        assert len(successful_results) >= 3  # At least 3 successful results
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_utilization(self, performance_config):
        """Test resource utilization during peak load."""
        crawler = NewsCrawler(config=performance_config)
        test_urls = [f'https://httpbin.org/json?resource={i}' for i in range(30)]
        
        # Monitor resource usage
        cpu_samples = []
        memory_samples = []
        
        def monitor_resources():
            process = psutil.Process()
            cpu_samples.append(process.cpu_percent())
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
        
        # Start monitoring
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.daemon = True
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                # Start monitoring
                monitor_thread.start()
                
                # Perform intensive crawling
                tasks = []
                for i in range(3):  # 3 concurrent batches
                    batch_urls = test_urls[i*10:(i+1)*10]
                    task = asyncio.create_task(crawler.crawl_urls(batch_urls))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                # Stop monitoring
                monitor_resources()
        
        # Analyze resource usage
        if cpu_samples:
            avg_cpu = statistics.mean(cpu_samples)
            max_cpu = max(cpu_samples)
            
            # CPU usage should be reasonable
            assert avg_cpu < 80  # Average CPU usage less than 80%
            assert max_cpu < 95  # Peak CPU usage less than 95%
        
        if memory_samples:
            memory_increase = max(memory_samples) - min(memory_samples)
            
            # Memory increase should be reasonable
            assert memory_increase < 200  # Less than 200MB increase
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, performance_config):
        """Test performance under sustained load."""
        crawler = NewsCrawler(config=performance_config)
        
        # Run for extended period
        total_requests = 0
        start_time = time.time()
        
        with patch.object(crawler, 'stealth_orchestrator') as mock_stealth:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth.fetch_with_stealth = AsyncMock(return_value=mock_response)
            
            async with crawler:
                # Run for 2 minutes with continuous requests
                end_time = start_time + 120  # 2 minutes
                
                while time.time() < end_time:
                    batch_urls = [f'https://httpbin.org/json?sustained={i}' for i in range(10)]
                    results = await crawler.crawl_urls(batch_urls)
                    total_requests += len(batch_urls)
                    
                    # Brief pause to prevent overwhelming
                    await asyncio.sleep(0.1)
        
        total_duration = time.time() - start_time
        average_throughput = total_requests / total_duration
        
        # Should maintain reasonable throughput
        assert average_throughput > 3  # At least 3 requests per second
        assert total_requests > 300  # Should process at least 300 requests
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_startup_shutdown_performance(self, performance_config):
        """Test startup and shutdown performance."""
        startup_times = []
        shutdown_times = []
        
        # Test multiple startup/shutdown cycles
        for _ in range(5):
            crawler = NewsCrawler(config=performance_config)
            
            # Measure startup time
            start_time = time.time()
            await crawler.start()
            startup_duration = time.time() - start_time
            startup_times.append(startup_duration)
            
            # Measure shutdown time
            start_time = time.time()
            await crawler.stop()
            shutdown_duration = time.time() - start_time
            shutdown_times.append(shutdown_duration)
        
        # Analyze startup/shutdown performance
        avg_startup = statistics.mean(startup_times)
        avg_shutdown = statistics.mean(shutdown_times)
        
        # Should start and stop quickly
        assert avg_startup < 2.0  # Less than 2 seconds to start
        assert avg_shutdown < 1.0  # Less than 1 second to stop
        
        # Should be consistent
        startup_std = statistics.stdev(startup_times)
        shutdown_std = statistics.stdev(shutdown_times)
        
        assert startup_std < 0.5  # Consistent startup times
        assert shutdown_std < 0.2  # Consistent shutdown times
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_instance_performance(self, performance_config):
        """Test performance with multiple crawler instances."""
        # Create multiple crawler instances
        crawlers = [NewsCrawler(config=performance_config) for _ in range(3)]
        
        test_urls = [f'https://httpbin.org/json?instance={i}' for i in range(15)]
        
        start_time = time.time()
        
        # Mock stealth orchestrator for all crawlers
        patches = []
        for crawler in crawlers:
            mock_stealth = patch.object(crawler, 'stealth_orchestrator')
            mock_stealth_obj = mock_stealth.start()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"test": "data"}'
            mock_stealth_obj.fetch_with_stealth = AsyncMock(return_value=mock_response)
            patches.append(mock_stealth)
        
        try:
            # Start all crawlers
            for crawler in crawlers:
                await crawler.start()
            
            # Run crawling tasks concurrently
            tasks = []
            for i, crawler in enumerate(crawlers):
                batch_urls = test_urls[i*5:(i+1)*5]
                task = asyncio.create_task(crawler.crawl_urls(batch_urls))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Stop all crawlers
            for crawler in crawlers:
                await crawler.stop()
        
        finally:
            # Clean up patches
            for patch_obj in patches:
                patch_obj.stop()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle multiple instances efficiently
        assert duration < 10  # Should complete within 10 seconds
        assert len(results) == 3  # One result per crawler
        
        # Check that all instances performed well
        for result in results:
            assert len(result) == 5  # Each crawler processed 5 URLs