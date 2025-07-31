"""
Integration Tests
================

Comprehensive integration tests for the gen_crawler package
testing real-world scenarios and component interactions.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

try:
    from ..core.gen_crawler import GenCrawler, GenSiteResult
    from ..core.adaptive_crawler import AdaptiveCrawler, CrawlStrategy
    from ..config.gen_config import create_gen_config
    from ..parsers.content_parser import GenContentParser
    from ..cli.exporters import MarkdownExporter, JSONExporter
    from .. import get_gen_crawler_health, crawl_site
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False

if not INTEGRATION_AVAILABLE:
    pytest.skip("Integration components not available", allow_module_level=True)

class TestPackageHealth(unittest.TestCase):
    """Test package health and availability."""
    
    def test_package_health_check(self):
        """Test package health monitoring."""
        health = get_gen_crawler_health()
        
        self.assertIsInstance(health, dict)
        self.assertIn('version', health)
        self.assertIn('status', health)
        self.assertIn('capabilities', health)
        
        # Version should be valid
        self.assertRegex(health['version'], r'\d+\.\d+\.\d+')
        
        # Status should be known
        self.assertIn(health['status'], ['healthy', 'degraded'])
        
        # Capabilities should be present
        capabilities = health['capabilities']
        self.assertIn('full_site_crawling', capabilities)
        self.assertIn('adaptive_crawling', capabilities)
        self.assertIn('content_analysis', capabilities)

class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration across components."""
    
    def test_config_to_crawler_integration(self):
        """Test configuration integration with crawler."""
        # Create custom configuration
        config_manager = create_gen_config()
        config_manager.settings.performance.max_pages_per_site = 100
        config_manager.settings.performance.max_concurrent = 3
        config_manager.settings.content_filters.min_content_length = 200
        
        # Get crawler configuration
        crawler_config = config_manager.get_crawler_config()
        
        # Verify configuration translation
        self.assertEqual(crawler_config['max_pages_per_site'], 100)
        self.assertEqual(crawler_config['max_concurrent'], 3)
        self.assertEqual(
            crawler_config['content_filters']['min_content_length'], 200
        )
        
        # Create crawler with configuration
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
            crawler = GenCrawler(crawler_config)
            
            self.assertEqual(crawler.config['max_pages_per_site'], 100)
            self.assertEqual(crawler.config['max_concurrent'], 3)
    
    def test_adaptive_config_integration(self):
        """Test adaptive configuration integration."""
        config_manager = create_gen_config()
        config_manager.settings.adaptive.enable_adaptive_crawling = False
        config_manager.settings.adaptive.strategy_switching_threshold = 0.9
        
        adaptive_config = config_manager.get_adaptive_config()
        
        self.assertFalse(adaptive_config['enable_adaptive_crawling'])
        self.assertEqual(adaptive_config['strategy_switching_threshold'], 0.9)
        
        # Test with adaptive crawler
        adaptive_crawler = AdaptiveCrawler(adaptive_config)
        
        # Should still work with custom config
        profile = adaptive_crawler.get_site_profile("https://example.com")
        self.assertIsNotNone(profile)

@pytest.mark.asyncio
class TestCrawlerWorkflow:
    """Test complete crawler workflow integration."""
    
    @pytest.fixture
    def test_html_content(self):
        """Sample HTML content for testing."""
        return """
        <html>
        <head>
            <title>Test Article: Important News</title>
            <meta name="author" content="Test Author">
            <meta name="description" content="Test article description">
        </head>
        <body>
            <article>
                <h1>Test Article: Important News</h1>
                <p>This is the first paragraph of our test article with substantial content.</p>
                <p>This is the second paragraph that provides more detailed information.</p>
                <p>The third paragraph continues the discussion with additional insights.</p>
                <a href="/related-article">Related Article</a>
                <a href="/another-story">Another Story</a>
            </article>
        </body>
        </html>
        """
    
    @pytest.fixture
    def mock_crawlee_environment(self):
        """Mock Crawlee environment for testing."""
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
            with patch('gen_crawler.core.gen_crawler.AdaptivePlaywrightCrawler') as mock_crawler_class:
                # Create mock crawler instance
                mock_crawler = MagicMock()
                mock_crawler.router.default_handler = MagicMock()
                mock_crawler.run = AsyncMock()
                mock_crawler_class.return_value = mock_crawler
                
                yield mock_crawler
    
    async def test_full_crawl_workflow(self, mock_crawlee_environment, test_html_content):
        """Test complete crawling workflow."""
        # Create configuration
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = 5
        config.settings.performance.max_concurrent = 2
        
        crawler_config = config.get_crawler_config()
        
        # Create crawler
        crawler = GenCrawler(crawler_config)
        await crawler.initialize()
        
        # Mock successful crawl
        mock_site_result = GenSiteResult(
            base_url="https://example.com",
            total_pages=3,
            successful_pages=3,
            failed_pages=0,
            total_time=5.0
        )
        
        # Add mock pages
        from ..core.gen_crawler import GenCrawlResult
        mock_pages = [
            GenCrawlResult(
                url="https://example.com/article1",
                title="Article 1",
                content="Content of article 1",
                word_count=50,
                success=True,
                content_type="article"
            ),
            GenCrawlResult(
                url="https://example.com/article2", 
                title="Article 2",
                content="Content of article 2",
                word_count=75,
                success=True,
                content_type="article"
            ),
            GenCrawlResult(
                url="https://example.com/page1",
                title="Page 1",
                content="Content of page 1",
                word_count=30,
                success=True,
                content_type="page"
            )
        ]
        mock_site_result.pages = mock_pages
        
        # Mock the crawl_site method
        with patch.object(crawler, 'crawl_site', return_value=mock_site_result):
            result = await crawler.crawl_site("https://example.com")
        
        # Verify results
        assert result.total_pages == 3
        assert result.successful_pages == 3
        assert result.success_rate == 100.0
        assert len(result.pages) == 3
        
        # Verify content types
        articles = result.get_articles()
        assert len(articles) == 2
        
        await crawler.cleanup()
    
    async def test_adaptive_strategy_integration(self, mock_crawlee_environment):
        """Test adaptive strategy integration with crawler."""
        # Create adaptive crawler
        adaptive = AdaptiveCrawler()
        
        # Create crawler with adaptive config
        config = create_gen_config()
        crawler_config = config.get_crawler_config()
        crawler = GenCrawler(crawler_config)
        
        # Test strategy recommendation
        url = "https://example.com"
        strategy = adaptive.recommend_strategy(url)
        
        assert strategy in [
            CrawlStrategy.ADAPTIVE,
            CrawlStrategy.HTTP_ONLY,
            CrawlStrategy.BROWSER_ONLY,
            CrawlStrategy.MIXED
        ]
        
        # Test performance update
        adaptive.update_strategy_performance(url, strategy, 85.0, 2.5)
        
        # Verify profile creation
        profile = adaptive.get_site_profile(url)
        assert profile.domain == "example.com"
        assert profile.total_pages > 0
    
    async def test_content_analysis_integration(self, test_html_content):
        """Test content analysis integration."""
        parser = GenContentParser()
        
        # Parse content
        parsed = parser.parse_content("https://example.com/article", test_html_content)
        
        # Verify parsing results
        assert "Test Article" in parsed.title
        assert "first paragraph" in parsed.content
        assert parsed.word_count > 0
        assert parsed.quality_score > 0
        assert parsed.content_type in ["article", "page", "content_page"]
        
        # Test quality assessment
        assert isinstance(parsed.is_article, bool)
        assert isinstance(parsed.is_high_quality, bool)

class TestExportIntegration(unittest.TestCase):
    """Test export functionality integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ..core.gen_crawler import GenCrawlResult, GenSiteResult
        
        # Create mock crawl results
        self.mock_pages = [
            GenCrawlResult(
                url="https://example.com/article1",
                title="First Article",
                content="Content of the first article with substantial text.",
                cleaned_content="Clean content of the first article.",
                word_count=50,
                success=True,
                content_type="article",
                quality_score=0.8,
                authors=["John Doe"],
                keywords=["test", "example"]
            ),
            GenCrawlResult(
                url="https://example.com/article2",
                title="Second Article", 
                content="Content of the second article with more text.",
                cleaned_content="Clean content of the second article.",
                word_count=75,
                success=True,
                content_type="article",
                quality_score=0.9,
                authors=["Jane Smith"],
                keywords=["sample", "demo"]
            )
        ]
        
        self.mock_site_result = GenSiteResult(
            base_url="https://example.com",
            pages=self.mock_pages,
            total_pages=2,
            successful_pages=2,
            failed_pages=0,
            total_time=10.0
        )
    
    def test_markdown_export_integration(self):
        """Test markdown export integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create markdown exporter
            exporter = MarkdownExporter(
                output_dir=output_dir,
                organize_by_site=True,
                include_metadata=True
            )
            
            # Export results
            asyncio.run(exporter.export_results([self.mock_site_result]))
            
            # Verify output structure
            site_dir = output_dir / "example_com"
            self.assertTrue(site_dir.exists())
            
            # Check for markdown files
            md_files = list(site_dir.glob("*.md"))
            self.assertEqual(len(md_files), 2)
            
            # Verify content
            for md_file in md_files:
                content = md_file.read_text(encoding='utf-8')
                self.assertIn("# ", content)  # Title
                self.assertIn("## Metadata", content)
                self.assertIn("## Content", content)
                self.assertIn("**URL**:", content)
                self.assertIn("**Word Count**:", content)
    
    def test_json_export_integration(self):
        """Test JSON export integration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Create JSON exporter
            exporter = JSONExporter(
                output_dir=output_dir,
                pretty_print=True
            )
            
            # Export results
            asyncio.run(exporter.export_results([self.mock_site_result]))
            
            # Verify output
            json_files = list(output_dir.glob("*.json"))
            self.assertGreater(len(json_files), 0)
            
            # Load and verify JSON
            json_file = json_files[0]
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.assertIn('results', data)
            self.assertEqual(len(data['results']), 1)
            
            site_data = data['results'][0]
            self.assertEqual(site_data['base_url'], "https://example.com")
            self.assertEqual(len(site_data['pages']), 2)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_crawler_without_crawlee(self):
        """Test crawler behavior without Crawlee."""
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', False):
            with self.assertRaises(ImportError):
                GenCrawler()
    
    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        from ..config.gen_config import GenCrawlerSettings, GenCrawlerConfig
        
        # Test invalid max_concurrent
        invalid_settings = GenCrawlerSettings()
        invalid_settings.performance.max_concurrent = -1
        
        with self.assertRaises(ValueError):
            GenCrawlerConfig(settings=invalid_settings)
    
    def test_parser_with_invalid_html(self):
        """Test parser with invalid HTML."""
        parser = GenContentParser()
        
        # Test with malformed HTML
        malformed_html = "<html><body><p>Unclosed paragraph<div>Nested incorrectly</p></div></body>"
        
        result = parser.parse_content("https://example.com/malformed", malformed_html)
        
        # Should not raise exception and should extract some content
        self.assertIsNotNone(result)
        self.assertEqual(result.url, "https://example.com/malformed")
        self.assertIsInstance(result.content, str)
    
    def test_export_with_empty_results(self):
        """Test export with empty results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            exporter = MarkdownExporter(output_dir=output_dir)
            
            # Should not raise exception
            asyncio.run(exporter.export_results([]))
            
            # Should create output directory but no content files
            self.assertTrue(output_dir.exists())

@pytest.mark.asyncio
class TestConvenienceFunctions:
    """Test convenience functions for easy usage."""
    
    async def test_crawl_site_convenience_function(self):
        """Test the crawl_site convenience function."""
        with patch('gen_crawler.CRAWLEE_AVAILABLE', True):
            with patch('gen_crawler.CORE_AVAILABLE', True):
                with patch('gen_crawler.GenCrawler') as mock_crawler_class:
                    # Mock crawler instance
                    mock_crawler = MagicMock()
                    mock_crawler.crawl_site = AsyncMock(return_value="mock_result")
                    mock_crawler.cleanup = AsyncMock()
                    mock_crawler_class.return_value = mock_crawler
                    
                    # Test convenience function
                    result = await crawl_site("https://example.com")
                    
                    assert result == "mock_result"
                    mock_crawler.crawl_site.assert_called_once_with("https://example.com")
                    mock_crawler.cleanup.assert_called_once()
    
    async def test_crawl_site_with_config(self):
        """Test crawl_site with custom configuration."""
        custom_config = {'max_pages_per_site': 10}
        
        with patch('gen_crawler.CRAWLEE_AVAILABLE', True):
            with patch('gen_crawler.CORE_AVAILABLE', True):
                with patch('gen_crawler.CONFIG_AVAILABLE', True):
                    with patch('gen_crawler.get_default_gen_config', return_value={}):
                        with patch('gen_crawler.GenCrawler') as mock_crawler_class:
                            mock_crawler = MagicMock()
                            mock_crawler.crawl_site = AsyncMock(return_value="result")
                            mock_crawler.cleanup = AsyncMock()
                            mock_crawler_class.return_value = mock_crawler
                            
                            result = await crawl_site("https://example.com", custom_config)
                            
                            # Verify crawler was created with merged config
                            mock_crawler_class.assert_called_once_with(custom_config)

class TestPerformanceIntegration(unittest.TestCase):
    """Test performance monitoring integration."""
    
    def test_performance_statistics_collection(self):
        """Test performance statistics collection."""
        # Create crawler with performance monitoring
        config = create_gen_config()
        config.settings.adaptive.performance_monitoring = True
        
        crawler_config = config.get_crawler_config()
        
        with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
            crawler = GenCrawler(crawler_config)
            
            # Get initial statistics
            stats = crawler.get_statistics()
            
            self.assertIsInstance(stats, dict)
            self.assertIn('pages_crawled', stats)
            self.assertIn('sites_crawled', stats)
            self.assertIn('config', stats)
            
            # Statistics should start at zero
            self.assertEqual(stats['pages_crawled'], 0)
            self.assertEqual(stats['sites_crawled'], 0)
    
    def test_adaptive_performance_tracking(self):
        """Test adaptive performance tracking."""
        adaptive = AdaptiveCrawler()
        
        # Simulate multiple crawl sessions
        test_urls = [
            "https://fast-site.com",
            "https://slow-site.com", 
            "https://medium-site.com"
        ]
        
        # Simulate different performance characteristics
        performance_data = [
            (CrawlStrategy.HTTP_ONLY, 95.0, 1.0),    # Fast site
            (CrawlStrategy.BROWSER_ONLY, 60.0, 5.0), # Slow site
            (CrawlStrategy.ADAPTIVE, 85.0, 2.5)      # Medium site
        ]
        
        for url, (strategy, success_rate, load_time) in zip(test_urls, performance_data):
            adaptive.update_strategy_performance(url, strategy, success_rate, load_time)
        
        # Verify statistics collection
        stats = adaptive.get_crawler_stats()
        
        self.assertEqual(stats['global_stats']['sites_analyzed'], 3)
        self.assertEqual(stats['global_stats']['total_crawls'], 3)
        
        # Verify strategy performance tracking
        strategy_perf = stats['global_stats']['strategy_performance']
        
        self.assertEqual(strategy_perf['http_only']['count'], 1)
        self.assertEqual(strategy_perf['browser_only']['count'], 1)
        self.assertEqual(strategy_perf['adaptive']['count'], 1)
        
        # Verify recommendations
        recommendations = stats['strategy_recommendations']
        self.assertIn('overall_best', recommendations)

class TestRealWorldScenarios(unittest.TestCase):
    """Test realistic usage scenarios."""
    
    def test_news_monitoring_scenario(self):
        """Test news monitoring scenario configuration."""
        # Create news-optimized configuration
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = 200
        config.settings.performance.crawl_delay = 3.0
        config.settings.content_filters.include_patterns = [
            'article', 'news', 'story', 'breaking'
        ]
        config.settings.content_filters.exclude_patterns = [
            'tag', 'category', 'archive', 'login', 'subscribe'
        ]
        
        crawler_config = config.get_crawler_config()
        
        # Verify news-optimized settings
        self.assertEqual(crawler_config['max_pages_per_site'], 200)
        self.assertEqual(crawler_config['crawl_delay'], 3.0)
        self.assertIn('article', crawler_config['content_filters']['include_patterns'])
        self.assertIn('tag', crawler_config['content_filters']['exclude_patterns'])
    
    def test_research_scenario(self):
        """Test research data collection scenario."""
        # Create research-optimized configuration  
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = 1000
        config.settings.content_filters.min_content_length = 500
        config.settings.adaptive.enable_adaptive_crawling = True
        config.settings.database.enable_database = True
        
        crawler_config = config.get_crawler_config()
        
        # Verify research-optimized settings
        self.assertEqual(crawler_config['max_pages_per_site'], 1000)
        self.assertEqual(crawler_config['content_filters']['min_content_length'], 500)
        self.assertTrue(crawler_config['enable_adaptive_crawling'])
        self.assertTrue(crawler_config['enable_database'])
    
    def test_conflict_monitoring_scenario(self):
        """Test conflict monitoring scenario."""
        # This would be used with CLI conflict keywords
        conflict_keywords = ['war', 'violence', 'crisis', 'protest', 'attack']
        
        # Create content for conflict analysis
        test_content = "Breaking news: Violent protests continue in the region amid ongoing crisis."
        
        parser = GenContentParser()
        analyzer = parser.analyzer
        
        # Simulate conflict detection logic
        content_lower = test_content.lower()
        conflict_matches = sum(
            1 for keyword in conflict_keywords
            if keyword in content_lower
        )
        
        self.assertGreater(conflict_matches, 0)
        
        # Verify multiple conflict keywords detected
        found_keywords = [
            kw for kw in conflict_keywords
            if kw in content_lower
        ]
        self.assertIn('violence', found_keywords)
        self.assertIn('crisis', found_keywords)

if __name__ == '__main__':
    # Run integration tests
    unittest.main(verbosity=2)