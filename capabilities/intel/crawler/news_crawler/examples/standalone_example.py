#!/usr/bin/env python3
"""
Standalone News Crawler Example
===============================

This example can be run independently and demonstrates how to use the 
comprehensive news crawler with all its features. It handles import 
paths automatically and provides fallbacks for missing components.

Usage:
    python standalone_example.py

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
current_dir = Path(__file__).parent
crawler_dir = current_dir.parent
packages_dir = crawler_dir.parent.parent.parent
sys.path.insert(0, str(packages_dir))

# Import news crawler components with multiple fallback strategies
CRAWLER_AVAILABLE = False
ComprehensiveNewsCrawler = None
CrawlTarget = None
create_comprehensive_crawler = None
create_comprehensive_config = None
EnhancedNewsCrawler = None

# Try different import strategies
import_strategies = [
    # Strategy 1: Direct package import
    lambda: __import__('lindela.packages_enhanced.crawlers.news_crawler.comprehensive_news_crawler', fromlist=['']),
    # Strategy 2: Relative import from parent
    lambda: __import__(f'{crawler_dir.name}.comprehensive_news_crawler', fromlist=['']),
    # Strategy 3: Local import
    lambda: __import__('comprehensive_news_crawler', fromlist=[''])
]

for strategy in import_strategies:
    try:
        comprehensive_module = strategy()
        ComprehensiveNewsCrawler = getattr(comprehensive_module, 'ComprehensiveNewsCrawler', None)
        CrawlTarget = getattr(comprehensive_module, 'CrawlTarget', None)
        create_comprehensive_crawler = getattr(comprehensive_module, 'create_comprehensive_crawler', None)
        create_comprehensive_config = getattr(comprehensive_module, 'create_comprehensive_config', None)
        
        if all([ComprehensiveNewsCrawler, CrawlTarget, create_comprehensive_crawler, create_comprehensive_config]):
            CRAWLER_AVAILABLE = True
            logger.info("Successfully imported comprehensive crawler components")
            break
    except ImportError as e:
        logger.debug(f"Import strategy failed: {e}")
        continue

# Try to import enhanced crawler
if not CRAWLER_AVAILABLE:
    try:
        # Try to import from the core module
        sys.path.insert(0, str(crawler_dir))
        from core.enhanced_news_crawler import EnhancedNewsCrawler, NewsArticle
        logger.info("Imported enhanced crawler as fallback")
        CRAWLER_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"Enhanced crawler import failed: {e}")

if not CRAWLER_AVAILABLE:
    logger.error("Could not import any crawler components")


class MockCrawlTarget:
    """Mock CrawlTarget for when the real one isn't available."""
    def __init__(self, url, name="", priority=1, max_pages=10):
        self.url = url
        self.name = name or url
        self.priority = priority
        self.max_pages = max_pages
        self.crawl_depth = 1
        self.follow_redirects = True
        self.custom_config = {}
        self.enabled = True


class MockNewsArticle:
    """Mock NewsArticle for when the real one isn't available."""
    def __init__(self, title="", content="", url="", **kwargs):
        self.title = title
        self.content = content
        self.url = url
        self.summary = kwargs.get('summary', '')
        self.authors = kwargs.get('authors', [])
        self.source_domain = kwargs.get('source_domain', '')
        self.quality_score = kwargs.get('quality_score', 0.0)
        self.ml_analysis = kwargs.get('ml_analysis', None)


class MockCrawlResults:
    """Mock CrawlResults for demonstration."""
    def __init__(self):
        self.session_id = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.targets_processed = 0
        self.articles_extracted = 0
        self.successful_extractions = 0
        self.failed_extractions = 0
        self.articles = []
        self.errors = []
        self.performance_metrics = {}
        self.bypass_stats = {}
    
    @property
    def success_rate(self):
        total = self.successful_extractions + self.failed_extractions
        return self.successful_extractions / max(total, 1)
    
    @property
    def duration(self):
        return self.end_time - self.start_time


class MockComprehensiveNewsCrawler:
    """Mock comprehensive crawler for demonstration purposes."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.targets = []
        self.session_id = f"mock_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Mock crawler initialized (session: {self.session_id})")
    
    def add_target(self, target):
        if isinstance(target, str):
            target = MockCrawlTarget(target)
        self.targets.append(target)
        logger.info(f"Added target: {target.name} ({target.url})")
    
    def add_targets(self, targets):
        for target in targets:
            self.add_target(target)
    
    async def crawl_all_targets(self, **kwargs):
        logger.info(f"Starting mock crawl of {len(self.targets)} targets")
        
        results = MockCrawlResults()
        results.targets_processed = len(self.targets)
        
        # Simulate crawling
        for i, target in enumerate(self.targets):
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Create mock article
            article = MockNewsArticle(
                title=f"Mock Article from {target.name}",
                content=f"This is mock content from {target.url}. " * 10,
                url=target.url,
                source_domain=target.name,
                quality_score=0.8,
                ml_analysis={
                    'sentiment_label': 'neutral',
                    'conflict_score': 0.2,
                    'entities': [{'text': 'Mock Entity', 'type': 'ORG'}],
                    'locations': ['Mock Location']
                }
            )
            
            results.articles.append(article)
            results.articles_extracted += 1
            results.successful_extractions += 1
            
            logger.info(f"Crawled target {i+1}/{len(self.targets)}: {target.name}")
        
        results.end_time = datetime.now()
        logger.info(f"Mock crawl completed: {results.articles_extracted} articles")
        
        return results
    
    async def crawl_single_url(self, url, **kwargs):
        logger.info(f"Mock crawling single URL: {url}")
        await asyncio.sleep(0.1)
        
        return MockNewsArticle(
            title=f"Mock Article from {url}",
            content="This is mock article content. " * 20,
            url=url,
            quality_score=0.75
        )
    
    def get_comprehensive_stats(self):
        return {
            'session_id': self.session_id,
            'configuration': self.config,
            'targets': {
                'total_targets': len(self.targets),
                'enabled_targets': len([t for t in self.targets if t.enabled])
            },
            'components': {
                'mock_mode': True,
                'real_crawler_available': CRAWLER_AVAILABLE
            }
        }
    
    async def cleanup(self):
        logger.info("Mock crawler cleanup completed")


# Use real components if available, otherwise use mocks
if not CRAWLER_AVAILABLE:
    logger.warning("Using mock crawler components for demonstration")
    ComprehensiveNewsCrawler = MockComprehensiveNewsCrawler
    CrawlTarget = MockCrawlTarget
    
    async def create_comprehensive_crawler(config=None):
        return MockComprehensiveNewsCrawler(config)
    
    def create_comprehensive_config(**kwargs):
        return {
            'enable_stealth': kwargs.get('enable_stealth', False),
            'enable_database': kwargs.get('enable_database', False),
            'max_concurrent': kwargs.get('max_concurrent', 5),
            'mock_mode': True
        }


async def basic_crawling_example():
    """Example: Basic news crawling with comprehensive crawler."""
    logger.info("=== Basic Crawling Example ===")
    
    try:
        # Create a basic crawler configuration
        config = create_comprehensive_config(
            enable_stealth=False,
            enable_database=False,
            max_concurrent=3
        )
        
        # Create the crawler
        crawler = await create_comprehensive_crawler(config)
        
        # Add some example targets
        targets = [
            CrawlTarget(
                url="https://httpbin.org/html",
                name="HTTPBin Test",
                priority=1,
                max_pages=1
            ),
            CrawlTarget(
                url="https://example.com",
                name="Example.com",
                priority=2,
                max_pages=1
            )
        ]
        
        # Add targets to crawler
        for target in targets:
            crawler.add_target(target)
        
        # Perform comprehensive crawl
        logger.info(f"Starting crawl of {len(targets)} targets...")
        results = await crawler.crawl_all_targets()
        
        # Display results
        logger.info(f"Crawl completed in {results.duration}")
        logger.info(f"Articles extracted: {results.articles_extracted}")
        logger.info(f"Success rate: {results.success_rate:.2%}")
        
        if results.articles:
            logger.info("Sample articles extracted:")
            for i, article in enumerate(results.articles[:3]):
                logger.info(f"  {i+1}. {article.title}")
                logger.info(f"     Content: {len(article.content)} characters")
                logger.info(f"     Quality: {article.quality_score:.2f}")
        
        # Get comprehensive stats
        stats = crawler.get_comprehensive_stats()
        logger.info(f"Crawler session: {stats['session_id']}")
        
        # Cleanup
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Basic crawling example failed: {e}")


async def single_url_example():
    """Example: Crawling a single URL."""
    logger.info("=== Single URL Crawling Example ===")
    
    try:
        config = create_comprehensive_config()
        crawler = await create_comprehensive_crawler(config)
        
        # Crawl a single URL
        test_url = "https://httpbin.org/html"
        logger.info(f"Crawling single URL: {test_url}")
        
        article = await crawler.crawl_single_url(test_url)
        
        if article:
            logger.info("Article extracted successfully:")
            logger.info(f"  Title: {article.title}")
            logger.info(f"  Content length: {len(article.content)} characters")
            logger.info(f"  Quality score: {article.quality_score:.2f}")
            
            if hasattr(article, 'ml_analysis') and article.ml_analysis:
                logger.info("ML Analysis results:")
                logger.info(f"  Sentiment: {article.ml_analysis.get('sentiment_label', 'N/A')}")
                logger.info(f"  Conflict score: {article.ml_analysis.get('conflict_score', 0):.2f}")
        else:
            logger.warning("Failed to extract article")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Single URL crawling example failed: {e}")


async def stealth_example():
    """Example: Stealth crawling demonstration."""
    logger.info("=== Stealth Crawling Example ===")
    
    try:
        # Create stealth configuration
        config = create_comprehensive_config(
            enable_stealth=True,
            enable_database=False,
            max_concurrent=2
        )
        
        crawler = await create_comprehensive_crawler(config)
        
        # Add targets that might benefit from stealth
        stealth_targets = [
            CrawlTarget(
                url="https://httpbin.org/user-agent",
                name="User Agent Test",
                priority=1,
                max_pages=1
            ),
            CrawlTarget(
                url="https://httpbin.org/headers",
                name="Headers Test",
                priority=1,
                max_pages=1
            )
        ]
        
        crawler.add_targets(stealth_targets)
        
        # Perform stealth crawl
        logger.info("Starting stealth crawl...")
        results = await crawler.crawl_all_targets()
        
        # Display stealth-specific results
        logger.info(f"Stealth crawl completed: {results.articles_extracted} articles")
        logger.info(f"Duration: {results.duration}")
        
        if hasattr(results, 'bypass_stats') and results.bypass_stats:
            logger.info("Bypass statistics available")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Stealth crawling example failed: {e}")


async def performance_example():
    """Example: Performance monitoring."""
    logger.info("=== Performance Monitoring Example ===")
    
    try:
        config = create_comprehensive_config(max_concurrent=5)
        crawler = await create_comprehensive_crawler(config)
        
        # Add multiple targets for performance testing
        targets = []
        for i in range(5):
            targets.append(CrawlTarget(
                url=f"https://httpbin.org/delay/{i % 3}",
                name=f"Performance Test {i}",
                priority=1,
                max_pages=1
            ))
        
        crawler.add_targets(targets)
        
        # Monitor performance
        start_time = datetime.now()
        results = await crawler.crawl_all_targets()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Performance Results:")
        logger.info(f"  Duration: {duration:.2f} seconds")
        logger.info(f"  Targets: {results.targets_processed}")
        logger.info(f"  Articles: {results.articles_extracted}")
        logger.info(f"  Throughput: {results.articles_extracted/max(duration, 1):.2f} articles/second")
        
        # Get comprehensive stats
        stats = crawler.get_comprehensive_stats()
        logger.info(f"Session ID: {stats['session_id']}")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Performance monitoring example failed: {e}")


async def save_results_example():
    """Example: Saving crawl results to files."""
    logger.info("=== Save Results Example ===")
    
    try:
        # Create output directory
        output_dir = Path("crawler_output")
        output_dir.mkdir(exist_ok=True)
        
        # Create crawler
        config = create_comprehensive_config()
        crawler = await create_comprehensive_crawler(config)
        
        # Add targets
        targets = [
            CrawlTarget(url="https://httpbin.org/html", name="Test Site 1"),
            CrawlTarget(url="https://example.com", name="Test Site 2")
        ]
        crawler.add_targets(targets)
        
        # Perform crawl
        results = await crawler.crawl_all_targets()
        
        # Save articles to JSON
        articles_data = []
        for article in results.articles:
            article_dict = {
                'title': article.title,
                'content': article.content,
                'url': article.url,
                'source_domain': getattr(article, 'source_domain', ''),
                'quality_score': getattr(article, 'quality_score', 0.0)
            }
            articles_data.append(article_dict)
        
        articles_file = output_dir / f"articles_{results.session_id}.json"
        with open(articles_file, 'w', encoding='utf-8') as f:
            json.dump(articles_data, f, indent=2, default=str)
        
        # Save crawl results summary
        results_summary = {
            'session_id': results.session_id,
            'start_time': results.start_time.isoformat(),
            'end_time': results.end_time.isoformat(),
            'duration_seconds': results.duration.total_seconds(),
            'targets_processed': results.targets_processed,
            'articles_extracted': results.articles_extracted,
            'success_rate': results.success_rate
        }
        
        summary_file = output_dir / f"summary_{results.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}/")
        logger.info(f"  Articles: {articles_file}")
        logger.info(f"  Summary: {summary_file}")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Save results example failed: {e}")


async def main():
    """Main function to run all examples."""
    logger.info("Starting News Crawler Examples")
    logger.info("=" * 50)
    
    if not CRAWLER_AVAILABLE:
        logger.warning("Real crawler components not available - using mock components")
        logger.warning("Install the full Lindela package for real functionality")
    
    examples = [
        ("Basic Crawling", basic_crawling_example),
        ("Single URL", single_url_example),
        ("Stealth Crawling", stealth_example),
        ("Performance Monitoring", performance_example),
        ("Save Results", save_results_example),
    ]
    
    for name, example_func in examples:
        try:
            logger.info(f"\n--- {name} ---")
            await example_func()
            logger.info(f"{name} completed successfully")
        except Exception as e:
            logger.error(f"{name} failed: {e}")
        
        logger.info("-" * 30)
        await asyncio.sleep(0.5)  # Brief pause between examples
    
    logger.info("All examples completed")


if __name__ == "__main__":
    print("News Crawler Standalone Example")
    print("================================")
    print()
    
    # Run the examples
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()