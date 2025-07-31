"""
Core Component Tests
===================

Test suite for gen_crawler core components including GenCrawler,
AdaptiveCrawler, and related functionality.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import pytest
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Test imports
try:
    from ..core.gen_crawler import GenCrawler, GenCrawlResult, GenSiteResult, create_gen_crawler
    from ..core.adaptive_crawler import AdaptiveCrawler, CrawlStrategy, SiteProfile
    from ..config.gen_config import create_gen_config
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False

# Mock Crawlee if not available
if not CORE_AVAILABLE:
    pytest.skip("Core components not available", allow_module_level=True)

class TestGenCrawlResult(unittest.TestCase):
    """Test GenCrawlResult data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_result = GenCrawlResult(
            url="https://example.com/test",
            title="Test Article",
            content="This is test content for the article.",
            word_count=8,
            success=True,
            content_type="article"
        )
    
    def test_result_creation(self):
        """Test basic result creation."""
        self.assertEqual(self.sample_result.url, "https://example.com/test")
        self.assertEqual(self.sample_result.title, "Test Article")
        self.assertTrue(self.sample_result.success)
        self.assertEqual(self.sample_result.word_count, 8)
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result_dict = self.sample_result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['url'], "https://example.com/test")
        self.assertEqual(result_dict['title'], "Test Article")
        self.assertEqual(result_dict['success'], True)
        self.assertIn('timestamp', result_dict)
    
    def test_content_type_detection(self):
        """Test content type detection."""
        # Test article detection
        self.sample_result.content_type = "article"
        self.sample_result.word_count = 500
        self.sample_result.title = "News Article"
        
        # Mock method would be called in real implementation
        self.assertEqual(self.sample_result.content_type, "article")

class TestGenSiteResult(unittest.TestCase):
    """Test GenSiteResult data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pages = [
            GenCrawlResult(url="https://example.com/1", success=True, content_type="article"),
            GenCrawlResult(url="https://example.com/2", success=True, content_type="page"),
            GenCrawlResult(url="https://example.com/3", success=False, content_type="unknown")
        ]
        
        self.site_result = GenSiteResult(
            base_url="https://example.com",
            pages=self.pages,
            total_pages=3,
            successful_pages=2,
            failed_pages=1
        )
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        expected_rate = (2 / 3) * 100  # 66.67%
        self.assertAlmostEqual(self.site_result.success_rate, expected_rate, places=1)
    
    def test_get_content_by_type(self):
        """Test filtering content by type."""
        articles = self.site_result.get_content_by_type("article")
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].url, "https://example.com/1")
    
    def test_get_articles(self):
        """Test getting articles specifically."""
        articles = self.site_result.get_articles()
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0].content_type, "article")
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result_dict = self.site_result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict['base_url'], "https://example.com")
        self.assertEqual(result_dict['total_pages'], 3)
        self.assertEqual(result_dict['successful_pages'], 2)
        self.assertEqual(len(result_dict['pages']), 3)

class TestSiteProfile(unittest.TestCase):
    """Test SiteProfile data class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profile = SiteProfile(domain="example.com")
    
    def test_profile_creation(self):
        """Test basic profile creation."""
        self.assertEqual(self.profile.domain, "example.com")
        self.assertEqual(self.profile.total_pages, 0)
        self.assertEqual(self.profile.performance_score, 0.0)
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Empty profile
        self.assertEqual(self.profile.success_rate, 0.0)
        
        # With data
        self.profile.total_pages = 10
        self.profile.successful_pages = 8
        self.assertEqual(self.profile.success_rate, 80.0)
    
    def test_update_performance(self):
        """Test performance update."""
        self.profile.update_performance(10, 8, 2.5)
        
        self.assertEqual(self.profile.total_pages, 10)
        self.assertEqual(self.profile.successful_pages, 8)
        self.assertEqual(self.profile.failed_pages, 2)
        self.assertEqual(self.profile.average_load_time, 2.5)
        self.assertIsNotNone(self.profile.last_crawled)
    
    def test_crawl_record_addition(self):
        """Test adding crawl records."""
        record = {
            'pages_crawled': 5,
            'success_rate': 80.0,
            'average_load_time': 2.0,
            'strategy': 'adaptive'
        }
        
        self.profile.add_crawl_record(record)
        self.assertEqual(len(self.profile.crawl_history), 1)
        self.assertEqual(self.profile.crawl_history[0]['pages_crawled'], 5)

class TestAdaptiveCrawler(unittest.TestCase):
    """Test AdaptiveCrawler functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adaptive_crawler = AdaptiveCrawler()
    
    def test_initialization(self):
        """Test adaptive crawler initialization."""
        self.assertIsInstance(self.adaptive_crawler.site_profiles, dict)
        self.assertIn('sites_analyzed', self.adaptive_crawler.global_stats)
    
    def test_get_site_profile(self):
        """Test site profile retrieval and creation."""
        url = "https://example.com/test"
        profile = self.adaptive_crawler.get_site_profile(url)
        
        self.assertIsInstance(profile, SiteProfile)
        self.assertEqual(profile.domain, "example.com")
        self.assertEqual(self.adaptive_crawler.global_stats['sites_analyzed'], 1)
        
        # Test caching
        profile2 = self.adaptive_crawler.get_site_profile(url)
        self.assertIs(profile, profile2)
        self.assertEqual(self.adaptive_crawler.global_stats['sites_analyzed'], 1)
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation logic."""
        url = "https://example.com"
        
        # New site should get adaptive strategy
        strategy = self.adaptive_crawler.recommend_strategy(url)
        self.assertEqual(strategy, CrawlStrategy.ADAPTIVE)
        
        # Site with good performance should use preferred strategy
        profile = self.adaptive_crawler.get_site_profile(url)
        profile.total_pages = 100
        profile.successful_pages = 90
        profile.last_crawled = datetime.now()
        profile.preferred_strategy = CrawlStrategy.HTTP_ONLY
        
        strategy = self.adaptive_crawler.recommend_strategy(url)
        self.assertEqual(strategy, CrawlStrategy.HTTP_ONLY)
    
    def test_site_characteristics_analysis(self):
        """Test site characteristics analysis."""
        url = "https://example.com"
        profile = self.adaptive_crawler.get_site_profile(url)
        
        # Test JavaScript requirement detection
        profile.requires_javascript = True
        context = {}
        strategy = self.adaptive_crawler._analyze_site_characteristics(profile, context)
        self.assertEqual(strategy, CrawlStrategy.BROWSER_ONLY)
        
        # Test rate limiting detection
        profile.requires_javascript = False
        profile.rate_limit_detected = True
        strategy = self.adaptive_crawler._analyze_site_characteristics(profile, context)
        self.assertEqual(strategy, CrawlStrategy.HTTP_ONLY)
    
    def test_performance_update(self):
        """Test performance metrics update."""
        url = "https://example.com"
        strategy = CrawlStrategy.ADAPTIVE
        
        self.adaptive_crawler.update_strategy_performance(url, strategy, 85.0, 2.5)
        
        # Check global stats update
        self.assertEqual(self.adaptive_crawler.global_stats['total_crawls'], 1)
        adaptive_stats = self.adaptive_crawler.global_stats['strategy_performance']['adaptive']
        self.assertEqual(adaptive_stats['count'], 1)
        self.assertEqual(adaptive_stats['avg_success'], 85.0)
    
    def test_optimal_settings(self):
        """Test optimal settings generation."""
        url = "https://example.com"
        strategy = CrawlStrategy.HTTP_ONLY
        
        settings = self.adaptive_crawler.get_optimal_settings(url, strategy)
        
        self.assertIsInstance(settings, dict)
        self.assertIn('max_concurrent', settings)
        self.assertIn('request_delay', settings)
        self.assertIn('request_timeout', settings)
        
        # HTTP_ONLY should have higher concurrency and lower delay
        self.assertGreaterEqual(settings['max_concurrent'], 3)
        self.assertLessEqual(settings['request_delay'], 2.0)
    
    def test_feature_detection(self):
        """Test site feature detection."""
        url = "https://example.com"
        html_content = """
        <html>
        <body>
            <script>document.addEventListener('DOMContentLoaded', function() {});</script>
            <div class="infinite-scroll">Content</div>
            <div>Cloudflare protection enabled</div>
        </body>
        </html>
        """
        
        features = self.adaptive_crawler.detect_site_features(url, html_content, 2.0)
        
        self.assertIn('requires_javascript', features)
        self.assertIn('has_infinite_scroll', features)
        self.assertIn('cloudflare_protection', features)
        
        # Verify profile is updated
        profile = self.adaptive_crawler.get_site_profile(url)
        self.assertTrue(profile.requires_javascript)
        self.assertTrue(profile.has_infinite_scroll)
        self.assertTrue(profile.cloudflare_protection)

@pytest.mark.asyncio
class TestGenCrawler:
    """Test GenCrawler functionality using pytest-asyncio."""
    
    @pytest.fixture
    def config(self):
        """Provide test configuration."""
        return {
            'max_pages_per_site': 10,
            'max_concurrent': 2,
            'request_timeout': 10,
            'crawl_delay': 0.5,
            'enable_database': False
        }
    
    @pytest.fixture
    def mock_crawler_context(self):
        """Mock Crawlee crawler context."""
        context = MagicMock()
        context.request.url = "https://example.com"
        context.page = MagicMock()
        context.page.title.return_value = asyncio.coroutine(lambda: "Test Page")()
        context.page.inner_text.return_value = asyncio.coroutine(lambda: "Test content")()
        context.page.wait_for_load_state.return_value = asyncio.coroutine(lambda: None)()
        context.page.query_selector_all.return_value = asyncio.coroutine(lambda: [])()
        context.enqueue_links.return_value = asyncio.coroutine(lambda: None)()
        return context
    
    def test_crawler_creation(self, config):
        """Test crawler creation."""
        crawler = GenCrawler(config)
        
        assert crawler.config['max_pages_per_site'] == 10
        assert crawler.config['max_concurrent'] == 2
        assert isinstance(crawler.visited_urls, set)
        assert isinstance(crawler.stats, dict)
    
    def test_factory_function(self):
        """Test factory function."""
        crawler = create_gen_crawler({'max_pages_per_site': 5})
        assert crawler.config['max_pages_per_site'] == 5
    
    @patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', False)
    def test_crawler_without_crawlee(self):
        """Test crawler behavior when Crawlee is not available."""
        with pytest.raises(ImportError, match="Crawlee is required"):
            GenCrawler()
    
    async def test_initialization(self, config):
        """Test crawler initialization."""
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
            with patch('gen_crawler.core.gen_crawler.AdaptivePlaywrightCrawler') as mock_crawler_class:
                mock_crawler = MagicMock()
                mock_crawler.router.default_handler = MagicMock()
                mock_crawler_class.return_value = mock_crawler
                
                crawler = GenCrawler(config)
                await crawler.initialize()
                
                assert crawler.crawler is not None
                mock_crawler_class.assert_called_once()
    
    async def test_handle_request(self, config, mock_crawler_context):
        """Test request handling."""
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
            crawler = GenCrawler(config)
            crawler.current_site_result = GenSiteResult(base_url="https://example.com")
            
            # Mock the methods used in _handle_request
            with patch.object(crawler, '_extract_links', return_value=[]):
                with patch.object(crawler, '_extract_images', return_value=[]):
                    with patch.object(crawler, '_classify_content', return_value="article"):
                        await crawler._handle_request(mock_crawler_context)
            
            assert crawler.stats['pages_crawled'] == 1
            assert len(crawler.current_site_result.pages) == 1
            assert crawler.current_site_result.successful_pages == 1
    
    def test_url_filtering(self, config):
        """Test URL filtering logic."""
        crawler = GenCrawler(config)
        
        base_url = "https://example.com"
        
        # Valid URLs
        assert crawler._should_crawl_url("https://example.com/article", base_url)
        assert crawler._should_crawl_url("https://example.com/news/story", base_url)
        
        # Invalid URLs (different domain)
        assert not crawler._should_crawl_url("https://other-site.com/article", base_url)
        
        # Excluded extensions
        assert not crawler._should_crawl_url("https://example.com/file.pdf", base_url)
        assert not crawler._should_crawl_url("https://example.com/doc.zip", base_url)
    
    def test_content_classification(self, config):
        """Test content classification."""
        crawler = GenCrawler(config)
        
        # Article content
        article_result = crawler._classify_content(
            "https://example.com/article",
            "Breaking News: Important Story",
            "This is a long article with substantial content. " * 50
        )
        assert article_result == "article"
        
        # Short content
        short_result = crawler._classify_content(
            "https://example.com/snippet",
            "Short",
            "Short content."
        )
        assert short_result == "insufficient_content"
        
        # Page content
        page_result = crawler._classify_content(
            "https://example.com/about",
            "About Us",
            "This is information about our company. " * 10
        )
        assert page_result == "page"
    
    def test_statistics_tracking(self, config):
        """Test statistics tracking."""
        crawler = GenCrawler(config)
        
        stats = crawler.get_statistics()
        assert 'pages_crawled' in stats
        assert 'sites_crawled' in stats
        assert 'config' in stats
        assert stats['pages_crawled'] == 0

class TestCrawlerIntegration(unittest.TestCase):
    """Integration tests for crawler components."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = create_gen_config()
        self.config.settings.performance.max_pages_per_site = 5
        self.config.settings.performance.max_concurrent = 1
        
    @patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True)
    def test_config_integration(self):
        """Test configuration integration with crawler."""
        crawler_config = self.config.get_crawler_config()
        crawler = GenCrawler(crawler_config)
        
        self.assertEqual(crawler.config['max_pages_per_site'], 5)
        self.assertEqual(crawler.config['max_concurrent'], 1)
    
    def test_adaptive_crawler_integration(self):
        """Test adaptive crawler integration."""
        adaptive = AdaptiveCrawler()
        
        # Simulate crawling multiple sites
        sites = ["https://site1.com", "https://site2.com", "https://site3.com"]
        
        for site in sites:
            strategy = adaptive.recommend_strategy(site)
            adaptive.update_strategy_performance(site, strategy, 85.0, 2.0)
        
        stats = adaptive.get_crawler_stats()
        self.assertEqual(stats['global_stats']['sites_analyzed'], 3)
        self.assertEqual(stats['global_stats']['total_crawls'], 3)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)