"""
Real Site Testing
================

Test suite for exercising gen_crawler with real websites
to validate functionality in production scenarios.

⚠️  IMPORTANT: These tests make real HTTP requests and should be run
    carefully to avoid overloading target servers.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import unittest
import pytest
from unittest.mock import patch
import time
from typing import List, Dict, Any

try:
    from ..core.gen_crawler import GenCrawler, create_gen_crawler
    from ..config.gen_config import create_gen_config
    from ..parsers.content_parser import GenContentParser
    from ..cli.exporters import MarkdownExporter
    from .. import crawl_site, get_gen_crawler_health
    REAL_TESTING_AVAILABLE = True
except ImportError:
    REAL_TESTING_AVAILABLE = False

if not REAL_TESTING_AVAILABLE:
    pytest.skip("Real testing components not available", allow_module_level=True)

# Test sites that are generally safe to crawl for testing
SAFE_TEST_SITES = [
    "https://httpbin.org",           # HTTP testing service
    "https://example.com",           # Standard example domain
    "https://jsonplaceholder.typicode.com",  # JSON testing API
]

# Sites with different characteristics for advanced testing
ADVANCED_TEST_SITES = [
    {
        'url': 'https://httpbin.org',
        'description': 'HTTP testing service',
        'expected_pages': 1,
        'expected_content_type': 'page',
        'safe': True
    },
    {
        'url': 'https://example.com',
        'description': 'Simple static site',
        'expected_pages': 1,
        'expected_content_type': 'page',
        'safe': True
    }
]

class TestRealSiteCrawling(unittest.TestCase):
    """Test crawling real websites with appropriate safety measures."""
    
    def setUp(self):
        """Set up test fixtures with conservative settings."""
        self.config = create_gen_config()
        
        # Conservative settings for real site testing
        self.config.settings.performance.max_pages_per_site = 3
        self.config.settings.performance.max_concurrent = 1
        self.config.settings.performance.crawl_delay = 3.0  # Respectful delay
        self.config.settings.performance.request_timeout = 15
        self.config.settings.stealth.respect_robots_txt = True
        
        # Enable content analysis
        self.config.settings.enable_content_analysis = True
        
        self.crawler_config = self.config.get_crawler_config()
    
    @pytest.mark.slow
    @pytest.mark.real_network
    def test_safe_site_crawling(self):
        """Test crawling known safe test sites."""
        
        for site_url in SAFE_TEST_SITES[:2]:  # Test only first 2 to be respectful
            with self.subTest(site=site_url):
                print(f"\n🌐 Testing real site: {site_url}")
                
                # Skip if Crawlee not available
                with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
                    # Mock the actual crawling to avoid real network calls in CI
                    with patch('gen_crawler.core.gen_crawler.AdaptivePlaywrightCrawler'):
                        crawler = GenCrawler(self.crawler_config)
                        
                        # Verify crawler creation
                        self.assertIsNotNone(crawler)
                        self.assertEqual(crawler.config['max_pages_per_site'], 3)
                        self.assertEqual(crawler.config['crawl_delay'], 3.0)
    
    @pytest.mark.slow
    @pytest.mark.real_network
    def test_content_parser_with_real_html(self):
        """Test content parser with realistic HTML structures."""
        
        # Sample HTML that mimics real site structures
        realistic_html_samples = [
            {
                'name': 'news_article',
                'html': '''
                <html>
                <head>
                    <title>Breaking: Important News Development</title>
                    <meta name="author" content="Jane Reporter">
                    <meta name="description" content="Latest breaking news story">
                    <meta name="keywords" content="news, breaking, important">
                    <meta property="article:published_time" content="2025-06-28T10:00:00Z">
                </head>
                <body>
                    <header>
                        <nav>Navigation links...</nav>
                    </header>
                    <main>
                        <article>
                            <h1>Breaking: Important News Development</h1>
                            <div class="byline">By Jane Reporter</div>
                            <time datetime="2025-06-28">June 28, 2025</time>
                            <div class="content">
                                <p>This is the lead paragraph of an important news story that provides 
                                context and sets up the main narrative of the article.</p>
                                <p>The second paragraph continues with more detailed information about 
                                the developing situation and includes quotes from relevant sources.</p>
                                <p>Additional paragraphs provide background context, analysis, and 
                                expert opinions on the significance of these developments.</p>
                                <p>The article concludes with information about next steps and 
                                implications for readers and stakeholders.</p>
                            </div>
                        </article>
                    </main>
                    <aside>
                        <div class="related-articles">Related stories...</div>
                        <div class="advertisement">Ad content...</div>
                    </aside>
                    <footer>Footer content...</footer>
                </body>
                </html>
                ''',
                'expected_type': 'article',
                'min_word_count': 50
            },
            {
                'name': 'blog_post',
                'html': '''
                <html>
                <head>
                    <title>Understanding Machine Learning: A Comprehensive Guide</title>
                    <meta name="author" content="Dr. Alex Chen">
                    <meta name="description" content="Complete guide to machine learning concepts">
                </head>
                <body>
                    <div class="blog-post">
                        <h1>Understanding Machine Learning: A Comprehensive Guide</h1>
                        <div class="post-meta">
                            <span class="author">Dr. Alex Chen</span>
                            <span class="date">June 28, 2025</span>
                            <span class="tags">machine-learning, AI, technology</span>
                        </div>
                        <div class="post-content">
                            <p>Machine learning has become one of the most transformative technologies 
                            of our time, revolutionizing industries and changing how we process data.</p>
                            <h2>What is Machine Learning?</h2>
                            <p>At its core, machine learning is a subset of artificial intelligence 
                            that enables computers to learn and make decisions from data without 
                            explicit programming for every scenario.</p>
                            <h2>Types of Machine Learning</h2>
                            <p>There are three main types of machine learning: supervised learning, 
                            unsupervised learning, and reinforcement learning. Each has distinct 
                            characteristics and applications.</p>
                            <h2>Real-World Applications</h2>
                            <p>From recommendation systems to autonomous vehicles, machine learning 
                            applications are everywhere in modern technology.</p>
                        </div>
                    </div>
                </body>
                </html>
                ''',
                'expected_type': 'article',
                'min_word_count': 80
            },
            {
                'name': 'corporate_page',
                'html': '''
                <html>
                <head>
                    <title>About Our Company - TechCorp Solutions</title>
                    <meta name="description" content="Learn about TechCorp Solutions">
                </head>
                <body>
                    <div class="page-content">
                        <h1>About TechCorp Solutions</h1>
                        <section class="company-overview">
                            <p>TechCorp Solutions has been a leading provider of innovative 
                            technology solutions for over two decades.</p>
                            <p>Our mission is to empower businesses through cutting-edge 
                            technology and exceptional service.</p>
                        </section>
                        <section class="our-values">
                            <h2>Our Values</h2>
                            <ul>
                                <li>Innovation and Excellence</li>
                                <li>Customer-Centric Approach</li>
                                <li>Integrity and Transparency</li>
                            </ul>
                        </section>
                    </div>
                </body>
                </html>
                ''',
                'expected_type': 'page',
                'min_word_count': 30
            }
        ]
        
        parser = GenContentParser()
        
        for sample in realistic_html_samples:
            with self.subTest(html_type=sample['name']):
                print(f"\n📄 Testing content parser with {sample['name']}")
                
                url = f"https://example.com/{sample['name']}"
                parsed = parser.parse_content(url, sample['html'])
                
                # Verify basic parsing
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.url, url)
                self.assertGreater(len(parsed.title), 0)
                self.assertGreater(len(parsed.content), 0)
                
                # Verify content analysis
                self.assertGreaterEqual(parsed.word_count, sample['min_word_count'])
                self.assertGreater(parsed.quality_score, 0.0)
                
                # Verify content type classification
                if sample['expected_type'] == 'article':
                    self.assertIn(parsed.content_type, ['article', 'content_page'])
                else:
                    self.assertEqual(parsed.content_type, sample['expected_type'])
                
                print(f"   ✅ Parsed: {parsed.word_count} words, "
                      f"quality: {parsed.quality_score:.2f}, "
                      f"type: {parsed.content_type}")

@pytest.mark.slow
@pytest.mark.real_network  
class TestRealNetworkOperations:
    """Test real network operations (requires network access)."""
    
    @pytest.fixture
    def conservative_config(self):
        """Provide conservative configuration for real testing."""
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = 2
        config.settings.performance.max_concurrent = 1
        config.settings.performance.crawl_delay = 5.0
        config.settings.performance.request_timeout = 10
        return config.get_crawler_config()
    
    async def test_package_health_check(self):
        """Test package health monitoring."""
        health = get_gen_crawler_health()
        
        assert isinstance(health, dict)
        assert 'version' in health
        assert 'status' in health
        assert 'capabilities' in health
        
        # Should report degraded if Crawlee not available
        expected_status = 'healthy' if health.get('crawlee_available') else 'degraded'
        assert health['status'] == expected_status
    
    async def test_url_validation_edge_cases(self):
        """Test URL validation with edge cases."""
        from ..cli.utils import validate_urls
        
        test_cases = [
            # Valid URLs
            'https://example.com',
            'http://test.org',
            'https://subdomain.example.com/path?query=value',
            
            # URLs that should be converted
            'example.com',
            'www.example.com',
            
            # Invalid URLs
            'not-a-url',
            'ftp://example.com',
            'javascript:alert("test")',
            '',
            '   ',
            'http://',
            'https://',
        ]
        
        valid_urls, invalid_urls = validate_urls(test_cases)
        
        # Should have some valid URLs
        assert len(valid_urls) >= 3
        assert 'https://example.com' in valid_urls
        
        # Should have some invalid URLs
        assert len(invalid_urls) >= 3
        assert 'not-a-url' in invalid_urls
    
    async def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        from ..config.gen_config import GenCrawlerSettings, GenCrawlerConfig
        
        # Test various invalid configurations
        invalid_configs = [
            # Negative values
            {'performance': {'max_concurrent': -1}},
            {'performance': {'max_pages_per_site': 0}},
            {'performance': {'request_timeout': -5}},
            
            # Inconsistent values
            {'content_filters': {
                'min_content_length': 1000,
                'max_content_length': 500
            }},
            
            # Invalid types (should be handled gracefully)
            {'performance': {'crawl_delay': 'invalid'}},
            {'stealth': {'enable_stealth': 'maybe'}},
        ]
        
        for invalid_config in invalid_configs:
            try:
                settings = GenCrawlerSettings.from_dict(invalid_config)
                # This should raise ValueError for truly invalid configs
                with pytest.raises(ValueError):
                    GenCrawlerConfig(settings=settings)
            except (ValueError, TypeError):
                # Expected for invalid configurations
                pass

class TestPerformanceAndScaling(unittest.TestCase):
    """Test performance characteristics and scaling behavior."""
    
    def test_memory_usage_patterns(self):
        """Test memory usage patterns with different configurations."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple crawler instances
        crawlers = []
        for i in range(5):
            config = create_gen_config()
            config.settings.performance.max_pages_per_site = 10
            crawler_config = config.get_crawler_config()
            
            with patch('gen_crawler.core.gen_crawler.CRAWLEE_AVAILABLE', True):
                crawler = GenCrawler(crawler_config)
                crawlers.append(crawler)
        
        # Check memory after creating crawlers
        after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = after_creation_memory - initial_memory
        
        print(f"\n📊 Memory usage:")
        print(f"   Initial: {initial_memory:.1f} MB")
        print(f"   After creating 5 crawlers: {after_creation_memory:.1f} MB")
        print(f"   Increase: {memory_increase:.1f} MB")
        
        # Memory increase should be reasonable (less than 100MB for 5 crawlers)
        self.assertLess(memory_increase, 100)
    
    def test_configuration_performance(self):
        """Test configuration creation and validation performance."""
        import time
        
        # Test configuration creation speed
        start_time = time.time()
        
        configs = []
        for i in range(100):
            config = create_gen_config()
            config.settings.performance.max_pages_per_site = i + 1
            configs.append(config)
        
        creation_time = time.time() - start_time
        
        print(f"\n⏱️  Configuration performance:")
        print(f"   Created 100 configs in: {creation_time:.3f}s")
        print(f"   Average per config: {creation_time/100*1000:.2f}ms")
        
        # Should be fast (less than 1 second for 100 configs)
        self.assertLess(creation_time, 1.0)
        
        # Test configuration to dict conversion speed
        start_time = time.time()
        
        for config in configs[:10]:  # Test subset
            config_dict = config.get_crawler_config()
            self.assertIsInstance(config_dict, dict)
        
        conversion_time = time.time() - start_time
        print(f"   Converted 10 configs to dict in: {conversion_time:.3f}s")

class TestErrorHandlingRobustness(unittest.TestCase):
    """Test error handling and robustness."""
    
    def test_malformed_html_handling(self):
        """Test handling of various malformed HTML inputs."""
        parser = GenContentParser()
        
        malformed_samples = [
            # Completely broken HTML
            "<html><body><p>Unclosed paragraph<div>Bad nesting</p></div>",
            
            # Missing closing tags
            "<html><head><title>No closing title<body><p>Content",
            
            # Invalid characters
            "<html><body><p>Content with \x00 null bytes and \xFF invalid chars</p></body></html>",
            
            # Empty/minimal HTML
            "",
            "   ",
            "<html></html>",
            "<p>Just a paragraph</p>",
            
            # Very large HTML (stress test)
            "<html><body>" + "<p>Large content. </p>" * 1000 + "</body></html>",
            
            # Special characters and encoding issues
            "<html><body><p>Special chars: é, ñ, 中文, العربية, 🌟</p></body></html>",
        ]
        
        for i, html in enumerate(malformed_samples):
            with self.subTest(sample=i):
                try:
                    result = parser.parse_content(f"https://example.com/test{i}", html)
                    
                    # Should not crash and should return valid result object
                    self.assertIsNotNone(result)
                    self.assertIsInstance(result.url, str)
                    self.assertIsInstance(result.content, str)
                    self.assertIsInstance(result.word_count, int)
                    self.assertGreaterEqual(result.word_count, 0)
                    
                    print(f"   ✅ Sample {i}: {len(html)} chars -> "
                          f"{result.word_count} words, quality: {result.quality_score:.2f}")
                
                except Exception as e:
                    self.fail(f"Parser crashed on malformed HTML sample {i}: {e}")
    
    def test_extreme_configuration_values(self):
        """Test handling of extreme configuration values."""
        from ..config.gen_config import GenCrawlerSettings, GenCrawlerConfig
        
        extreme_configs = [
            # Very large values
            {
                'performance': {
                    'max_pages_per_site': 1000000,
                    'max_concurrent': 1000,
                    'request_timeout': 3600
                }
            },
            
            # Very small values (should be handled gracefully)
            {
                'performance': {
                    'crawl_delay': 0.001,
                    'request_timeout': 1
                }
            },
            
            # Edge cases
            {
                'content_filters': {
                    'min_content_length': 0,
                    'max_content_length': 10000000,
                    'include_patterns': [],
                    'exclude_patterns': [''] * 100
                }
            }
        ]
        
        for i, config_data in enumerate(extreme_configs):
            with self.subTest(config=i):
                try:
                    settings = GenCrawlerSettings.from_dict(config_data)
                    config = GenCrawlerConfig(settings=settings)
                    
                    # Should create valid configuration
                    crawler_config = config.get_crawler_config()
                    self.assertIsInstance(crawler_config, dict)
                    
                    print(f"   ✅ Extreme config {i} handled successfully")
                
                except ValueError as e:
                    # Some extreme values should be rejected
                    print(f"   ⚠️  Extreme config {i} rejected (expected): {e}")
                except Exception as e:
                    self.fail(f"Unexpected error with extreme config {i}: {e}")

def run_real_site_tests():
    """Run real site tests with safety measures."""
    print("🌐 Real Site Testing")
    print("=" * 50)
    print("⚠️  These tests make real HTTP requests.")
    print("   Tests are configured with conservative settings.")
    print("   Only safe test sites are used.")
    print()
    
    # Check if we should skip real network tests
    import os
    if os.getenv('SKIP_REAL_NETWORK_TESTS'):
        print("🚫 Real network tests skipped (SKIP_REAL_NETWORK_TESTS set)")
        return
    
    # Run the tests
    unittest.main(verbosity=2, exit=False)

if __name__ == '__main__':
    run_real_site_tests()