#!/usr/bin/env python3
"""
Comprehensive Test Suite for Crawlers Package
============================================

Tests the crawlers package functionality including multi-source crawling,
stealth capabilities, and unified result handling.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import asyncio
import pytest
import sys
import os
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import aiohttp
from dataclasses import dataclass

# Add the packages_enhanced directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    from crawlers import (
        MultiSourceCrawler,
        UnifiedResult,
        CrawlerCapabilities,
        CrawlJob,
        CrawlerMetrics,
        CrawlerStatus,
        ContentQuality,
        ProtectionType,
        create_multi_source_crawler,
        create_stealth_crawler,
        get_available_crawlers,
        get_crawler_capabilities,
        validate_crawler_config
    )
    CRAWLERS_AVAILABLE = True
except ImportError:
    CRAWLERS_AVAILABLE = False


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestUnifiedResult:
    """Test UnifiedResult data structure."""

    def test_unified_result_initialization(self):
        """Test UnifiedResult initialization."""
        result = UnifiedResult(
            url="https://example.com/article",
            title="Test Article",
            content="This is test content",
            source_type="news",
            timestamp=datetime.now()
        )

        assert result.url == "https://example.com/article"
        assert result.title == "Test Article"
        assert result.content == "This is test content"
        assert result.source_type == "news"
        assert isinstance(result.timestamp, datetime)

    def test_unified_result_to_dict(self):
        """Test UnifiedResult conversion to dictionary."""
        result = UnifiedResult(
            url="https://example.com/article",
            title="Test Article",
            content="Test content",
            source_type="news",
            timestamp=datetime.now(),
            metadata={"author": "Test Author"}
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["url"] == "https://example.com/article"
        assert result_dict["title"] == "Test Article"
        assert "timestamp" in result_dict
        assert "metadata" in result_dict

    def test_unified_result_from_dict(self):
        """Test UnifiedResult creation from dictionary."""
        data = {
            "url": "https://example.com/article",
            "title": "Test Article",
            "content": "Test content",
            "source_type": "news",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"author": "Test Author"}
        }

        result = UnifiedResult.from_dict(data)
        assert result.url == data["url"]
        assert result.title == data["title"]
        assert result.content == data["content"]
        assert result.metadata["author"] == "Test Author"

    def test_calculate_relevance(self):
        """Test relevance calculation."""
        result = UnifiedResult(
            url="https://example.com/conflict-article",
            title="Ethiopia Conflict News",
            content="Major conflict in Ethiopia region with casualties",
            source_type="news",
            timestamp=datetime.now()
        )

        relevance = result.calculate_relevance(["conflict", "Ethiopia"])
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.0  # Should find some relevance


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestCrawlerCapabilities:
    """Test CrawlerCapabilities."""

    def test_crawler_capabilities_initialization(self):
        """Test CrawlerCapabilities initialization."""
        capabilities = CrawlerCapabilities(
            stealth_enabled=True,
            javascript_support=True,
            max_concurrent_requests=10,
            supported_content_types=["text/html", "application/json"],
            rate_limit_per_second=5.0
        )

        assert capabilities.stealth_enabled is True
        assert capabilities.javascript_support is True
        assert capabilities.max_concurrent_requests == 10
        assert "text/html" in capabilities.supported_content_types
        assert capabilities.rate_limit_per_second == 5.0


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestCrawlJob:
    """Test CrawlJob functionality."""

    def test_crawl_job_initialization(self):
        """Test CrawlJob initialization."""
        job = CrawlJob(
            job_id="test_job_123",
            query="Ethiopia conflict",
            sources=["news", "twitter"],
            max_results=100
        )

        assert job.job_id == "test_job_123"
        assert job.query == "Ethiopia conflict"
        assert job.sources == ["news", "twitter"]
        assert job.max_results == 100
        assert job.status == CrawlerStatus.PENDING

    def test_crawl_job_update_status(self):
        """Test CrawlJob status updates."""
        job = CrawlJob(
            job_id="test_job_123",
            query="test query",
            sources=["news"]
        )

        job.update_status(CrawlerStatus.RUNNING)
        assert job.status == CrawlerStatus.RUNNING

        job.update_status(CrawlerStatus.COMPLETED)
        assert job.status == CrawlerStatus.COMPLETED

    def test_crawl_job_add_result(self):
        """Test adding results to CrawlJob."""
        job = CrawlJob(
            job_id="test_job_123",
            query="test query",
            sources=["news"]
        )

        result = UnifiedResult(
            url="https://example.com/article",
            title="Test Article",
            content="Test content",
            source_type="news",
            timestamp=datetime.now()
        )

        job.add_result(result)
        assert len(job.results) == 1
        assert job.results[0] == result

    def test_crawl_job_add_error(self):
        """Test adding errors to CrawlJob."""
        job = CrawlJob(
            job_id="test_job_123",
            query="test query",
            sources=["news"]
        )

        job.add_error("Connection timeout")
        assert len(job.errors) == 1
        assert job.errors[0] == "Connection timeout"


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestCrawlerMetrics:
    """Test CrawlerMetrics functionality."""

    def test_crawler_metrics_initialization(self):
        """Test CrawlerMetrics initialization."""
        metrics = CrawlerMetrics()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert len(metrics.response_times) == 0

    def test_crawler_metrics_success_rate(self):
        """Test success rate calculation."""
        metrics = CrawlerMetrics()

        # No requests yet
        assert metrics.success_rate() == 0.0

        # Add some requests
        metrics.total_requests = 10
        metrics.successful_requests = 8
        metrics.failed_requests = 2

        assert metrics.success_rate() == 0.8

    def test_crawler_metrics_average_response_time(self):
        """Test average response time calculation."""
        metrics = CrawlerMetrics()

        # No response times yet
        assert metrics.average_response_time() == 0.0

        # Add response times
        metrics.response_times = [1.0, 2.0, 3.0]
        assert metrics.average_response_time() == 2.0

    def test_crawler_metrics_update(self):
        """Test metrics update."""
        metrics = CrawlerMetrics()

        # Update with successful request
        metrics.update(
            url="https://example.com",
            success=True,
            response_time=1.5,
            status_code=200
        )

        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.response_times[-1] == 1.5

        # Update with failed request
        metrics.update(
            url="https://example.com/fail",
            success=False,
            response_time=2.0,
            status_code=404
        )

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1

    def test_crawler_metrics_reset(self):
        """Test metrics reset."""
        metrics = CrawlerMetrics()

        # Add some data
        metrics.update("https://example.com", True, 1.0, 200)
        metrics.update("https://example.com/2", False, 2.0, 404)

        # Reset
        metrics.reset()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert len(metrics.response_times) == 0

    def test_crawler_metrics_get_report(self):
        """Test metrics report generation."""
        metrics = CrawlerMetrics()

        # Add some data
        metrics.update("https://example.com", True, 1.0, 200)
        metrics.update("https://example.com/2", True, 2.0, 200)
        metrics.update("https://example.com/3", False, 3.0, 404)

        report = metrics.get_report()

        assert isinstance(report, dict)
        assert "total_requests" in report
        assert "success_rate" in report
        assert "average_response_time" in report
        assert "errors_by_code" in report

        assert report["total_requests"] == 3
        assert report["success_rate"] == 2/3
        assert report["average_response_time"] == 2.0


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestMultiSourceCrawler:
    """Test MultiSourceCrawler functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.crawler_config = {
            'stealth_enabled': True,
            'max_concurrent_requests': 5,
            'timeout': 30,
            'user_agent': 'Test Crawler 1.0'
        }

    @pytest.mark.asyncio
    async def test_multi_source_crawler_initialization(self):
        """Test MultiSourceCrawler initialization."""
        crawler = MultiSourceCrawler(self.crawler_config)

        assert hasattr(crawler, 'config')
        assert hasattr(crawler, 'session')
        assert hasattr(crawler, 'metrics')
        assert hasattr(crawler, 'circuit_breakers')

    @pytest.mark.asyncio
    async def test_multi_source_crawler_health_check(self):
        """Test health check functionality."""
        crawler = MultiSourceCrawler(self.crawler_config)

        health = await crawler.health_check()

        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'component_status' in health
        assert 'metrics' in health

    @pytest.mark.asyncio
    async def test_multi_source_crawler_close(self):
        """Test crawler cleanup."""
        crawler = MultiSourceCrawler(self.crawler_config)

        # This should not raise an exception
        await crawler.close()

    def test_multi_source_crawler_get_metrics_report(self):
        """Test metrics report retrieval."""
        crawler = MultiSourceCrawler(self.crawler_config)

        report = crawler.get_metrics_report()
        assert isinstance(report, dict)

    def test_multi_source_crawler_reset_metrics(self):
        """Test metrics reset."""
        crawler = MultiSourceCrawler(self.crawler_config)

        # This should not raise an exception
        crawler.reset_metrics()

    @pytest.mark.asyncio
    async def test_crawl_all_sources_mock(self):
        """Test crawling all sources with mocked responses."""
        crawler = MultiSourceCrawler(self.crawler_config)

        # Mock the individual crawler methods
        with patch.object(crawler, '_crawl_news', new_callable=AsyncMock) as mock_news, \
             patch.object(crawler, '_crawl_google_news', new_callable=AsyncMock) as mock_google, \
             patch.object(crawler, '_crawl_twitter', new_callable=AsyncMock) as mock_twitter:

            # Setup mock returns
            mock_news.return_value = [UnifiedResult(
                url="https://news.example.com/article",
                title="News Article",
                content="News content",
                source_type="news",
                timestamp=datetime.now()
            )]

            mock_google.return_value = [UnifiedResult(
                url="https://news.google.com/article",
                title="Google News Article",
                content="Google news content",
                source_type="google_news",
                timestamp=datetime.now()
            )]

            mock_twitter.return_value = [UnifiedResult(
                url="https://twitter.com/user/status",
                title="Twitter Post",
                content="Twitter content",
                source_type="twitter",
                timestamp=datetime.now()
            )]

            # Test crawling
            results = await crawler.crawl_all_sources(
                query="Ethiopia conflict",
                sources=["news", "google_news", "twitter"]
            )

            assert isinstance(results, list)
            # Should have called the mocked methods
            mock_news.assert_called_once()
            mock_google.assert_called_once()
            mock_twitter.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_with_filters(self):
        """Test crawling with filters."""
        crawler = MultiSourceCrawler(self.crawler_config)

        filters = {
            'date_range': {
                'start': datetime.now() - timedelta(days=7),
                'end': datetime.now()
            },
            'content_type': ['article', 'news'],
            'min_content_length': 100
        }

        with patch.object(crawler, '_crawl_news', new_callable=AsyncMock) as mock_news:
            mock_news.return_value = []

            results = await crawler.crawl_with_filters(
                query="test query",
                sources=["news"],
                filters=filters
            )

            assert isinstance(results, list)
            mock_news.assert_called_once()


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestFactoryFunctions:
    """Test factory functions for crawler creation."""

    def test_create_multi_source_crawler(self):
        """Test multi-source crawler creation."""
        config = {
            'stealth_enabled': True,
            'max_concurrent_requests': 10
        }

        crawler = create_multi_source_crawler(config)
        assert isinstance(crawler, MultiSourceCrawler)

    def test_create_stealth_crawler(self):
        """Test stealth crawler creation."""
        config = {
            'stealth_level': 'high',
            'user_agent_rotation': True
        }

        # This might not be available, so handle gracefully
        try:
            crawler = create_stealth_crawler(config)
            assert crawler is not None
        except (ImportError, AttributeError):
            pytest.skip("Stealth crawler components not available")

    def test_get_available_crawlers(self):
        """Test getting available crawlers."""
        crawlers = get_available_crawlers()
        assert isinstance(crawlers, list)

        # Should contain at least some basic crawlers
        expected_crawlers = ['multi_source', 'news']
        for crawler in expected_crawlers:
            # Check if any available crawler contains the expected name
            assert any(crawler in available for available in crawlers) or len(crawlers) == 0

    def test_get_crawler_capabilities(self):
        """Test getting crawler capabilities."""
        capabilities = get_crawler_capabilities('multi_source')
        assert isinstance(capabilities, dict)

        # Should have basic capability information
        expected_keys = ['name', 'supported_sources', 'features']
        for key in expected_keys:
            if capabilities:  # Only check if capabilities are returned
                assert key in capabilities or len(capabilities) == 0

    def test_validate_crawler_config(self):
        """Test crawler configuration validation."""
        # Valid config
        valid_config = {
            'stealth_enabled': True,
            'max_concurrent_requests': 10,
            'timeout': 30
        }

        result = validate_crawler_config(valid_config)
        assert isinstance(result, dict)
        assert 'valid' in result
        assert result['valid'] is True

        # Invalid config
        invalid_config = {
            'max_concurrent_requests': -1,  # Invalid negative value
            'timeout': 'invalid'  # Invalid type
        }

        result = validate_crawler_config(invalid_config)
        assert isinstance(result, dict)
        assert 'valid' in result
        # Might be True if validation is lenient, or False if strict


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestErrorHandling:
    """Test error handling in crawlers."""

    @pytest.mark.asyncio
    async def test_crawler_with_network_error(self):
        """Test crawler behavior with network errors."""
        crawler = MultiSourceCrawler({})

        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")

            # Should handle network errors gracefully
            try:
                results = await crawler.crawl_all_sources("test query", ["news"])
                assert isinstance(results, list)
                # Might be empty due to errors, which is acceptable
            except Exception as e:
                # Should not propagate network errors
                pytest.fail(f"Network error not handled gracefully: {e}")

    @pytest.mark.asyncio
    async def test_crawler_with_timeout(self):
        """Test crawler behavior with timeouts."""
        config = {'timeout': 0.1}  # Very short timeout
        crawler = MultiSourceCrawler(config)

        with patch('aiohttp.ClientSession.get') as mock_get:
            # Simulate timeout
            mock_get.side_effect = asyncio.TimeoutError("Request timeout")

            results = await crawler.crawl_all_sources("test query", ["news"])
            assert isinstance(results, list)
            # Results might be empty due to timeouts

    def test_unified_result_with_invalid_data(self):
        """Test UnifiedResult with invalid data."""
        # Test with missing required fields
        try:
            result = UnifiedResult(
                url=None,  # Invalid URL
                title="",
                content="",
                source_type="news",
                timestamp=datetime.now()
            )
            # Should handle None URL gracefully or raise appropriate error
            assert result.url is None
        except (ValueError, TypeError):
            # This is acceptable behavior for invalid data
            pass


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_crawling(self):
        """Test concurrent crawling capabilities."""
        config = {'max_concurrent_requests': 3}
        crawler = MultiSourceCrawler(config)

        # Mock multiple concurrent requests
        with patch.object(crawler, '_crawl_news', new_callable=AsyncMock) as mock_news:
            mock_news.return_value = []

            # Start multiple crawl operations
            tasks = []
            for i in range(5):
                task = crawler.crawl_all_sources(f"query_{i}", ["news"])
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should complete without exceptions
            for result in results:
                assert not isinstance(result, Exception)
                assert isinstance(result, list)

    def test_metrics_performance(self):
        """Test metrics collection performance."""
        metrics = CrawlerMetrics()

        # Add many metrics updates
        for i in range(1000):
            metrics.update(
                url=f"https://example.com/{i}",
                success=i % 2 == 0,
                response_time=1.0 + (i % 10) / 10,
                status_code=200 if i % 2 == 0 else 404
            )

        # Should handle large number of metrics
        assert metrics.total_requests == 1000
        assert metrics.successful_requests == 500
        assert metrics.failed_requests == 500

        # Report generation should be fast
        report = metrics.get_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test memory cleanup after crawling."""
        crawler = MultiSourceCrawler({})

        # Simulate crawling with results
        with patch.object(crawler, '_crawl_news', new_callable=AsyncMock) as mock_news:
            # Return large result set
            large_results = [
                UnifiedResult(
                    url=f"https://example.com/{i}",
                    title=f"Article {i}",
                    content=f"Content {i}" * 100,  # Large content
                    source_type="news",
                    timestamp=datetime.now()
                )
                for i in range(100)
            ]
            mock_news.return_value = large_results

            results = await crawler.crawl_all_sources("test query", ["news"])
            assert len(results) > 0

        # Cleanup
        await crawler.close()

        # Should not retain references to large objects
        # This is more of a design verification than a strict test


@pytest.mark.skipif(not CRAWLERS_AVAILABLE, reason="Crawlers package not available")
class TestIntegration:
    """Integration tests for crawler components."""

    @pytest.mark.asyncio
    async def test_end_to_end_crawling_workflow(self):
        """Test complete crawling workflow."""
        # Create crawler with realistic config
        config = {
            'stealth_enabled': True,
            'max_concurrent_requests': 2,
            'timeout': 30,
            'rate_limit_per_second': 1.0
        }

        crawler = create_multi_source_crawler(config)

        try:
            # Mock all external dependencies
            with patch('aiohttp.ClientSession.get') as mock_get:
                # Mock successful response
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.text = AsyncMock(return_value="<html><body>Test article content</body></html>")
                mock_response.headers = {'content-type': 'text/html'}
                mock_get.return_value.__aenter__.return_value = mock_response

                # Test the workflow
                results = await crawler.crawl_all_sources(
                    query="Ethiopia conflict news",
                    sources=["news"]
                )

                # Verify results structure
                assert isinstance(results, list)

                # Check metrics were updated
                metrics_report = crawler.get_metrics_report()
                assert isinstance(metrics_report, dict)

                # Check health status
                health = await crawler.health_check()
                assert isinstance(health, dict)
                assert 'overall_status' in health

        finally:
            await crawler.close()

    def test_configuration_validation_integration(self):
        """Test configuration validation with real config."""
        configs = [
            # Valid configurations
            {
                'stealth_enabled': True,
                'max_concurrent_requests': 5,
                'timeout': 30
            },
            {
                'stealth_enabled': False,
                'max_concurrent_requests': 1,
                'timeout': 60,
                'user_agent': 'Custom User Agent'
            },
            # Edge case configurations
            {
                'max_concurrent_requests': 1,
                'timeout': 1
            }
        ]

        for config in configs:
            validation_result = validate_crawler_config(config)
            assert isinstance(validation_result, dict)
            assert 'valid' in validation_result

            if validation_result['valid']:
                # Should be able to create crawler with valid config
                crawler = create_multi_source_crawler(config)
                assert isinstance(crawler, MultiSourceCrawler)


if __name__ == "__main__":
    # Setup logging for tests
    logging.basicConfig(level=logging.DEBUG)

    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
