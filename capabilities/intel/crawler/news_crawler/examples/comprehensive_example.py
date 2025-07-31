"""
Comprehensive News Crawler Example
==================================

This example demonstrates how to use the comprehensive news crawler
with all its features including database integration, ML analysis,
stealth capabilities, and performance monitoring.

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

import asyncio
import logging
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import news crawler components
try:
    from ..comprehensive_news_crawler import (
        ComprehensiveNewsCrawler,
        CrawlTarget,
        create_comprehensive_crawler,
        create_comprehensive_config
    )
    from ..core import EnhancedNewsCrawler
    CRAWLER_AVAILABLE = True
except ImportError as e:
    logger.error(f"News crawler not available: {e}")
    CRAWLER_AVAILABLE = False


async def basic_crawling_example():
    """
    Example: Basic news crawling with comprehensive crawler.
    """
    logger.info("=== Basic Crawling Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
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
                logger.info(f"  {i+1}. {article.title[:50]}...")
        
        # Get comprehensive stats
        stats = crawler.get_comprehensive_stats()
        logger.info(f"Crawler session: {stats['session_id']}")
        logger.info(f"Components available: {sum(stats['components'].values())}/7")
        
        # Cleanup
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Basic crawling example failed: {e}")


async def stealth_crawling_example():
    """
    Example: Stealth crawling with bypass capabilities.
    """
    logger.info("=== Stealth Crawling Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
    try:
        # Create stealth configuration
        config = create_comprehensive_config(
            enable_stealth=True,
            enable_database=False,
            max_concurrent=2
        )
        
        # Add stealth-specific settings
        config.update({
            'stealth_config': {
                'stealth_level': 'high',
                'randomize_user_agent': True,
                'randomize_headers': True,
                'enable_proxy_rotation': False  # Set to True if you have proxies
            }
        })
        
        # Create the stealth crawler
        crawler = await create_comprehensive_crawler(config)
        
        # Add targets that might require stealth
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
        
        if results.bypass_stats:
            logger.info("Bypass statistics:")
            bypass_stats = results.bypass_stats.get('metrics', {})
            logger.info(f"  Total requests: {bypass_stats.get('total_requests', 0)}")
            logger.info(f"  Success rate: {bypass_stats.get('success_rate', 0):.2%}")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Stealth crawling example failed: {e}")


async def database_integration_example():
    """
    Example: News crawling with database integration.
    """
    logger.info("=== Database Integration Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
    try:
        # Note: This requires a PostgreSQL database
        # Update the connection string with your database details
        DATABASE_URL = "postgresql://user:password@localhost:5432/news_db"
        
        # Create configuration with database enabled
        config = create_comprehensive_config(
            enable_stealth=False,
            enable_database=True,  # Enable database integration
            max_concurrent=3
        )
        
        # Add database connection
        config['database_connection_string'] = DATABASE_URL
        
        # Create crawler with database
        crawler = await create_comprehensive_crawler(config)
        
        # Add news targets
        news_targets = [
            CrawlTarget(
                url="https://httpbin.org/html",
                name="Sample News Site",
                priority=1,
                max_pages=2
            )
        ]
        
        crawler.add_targets(news_targets)
        
        # Perform crawl with database storage
        logger.info("Starting crawl with database storage...")
        results = await crawler.crawl_all_targets()
        
        logger.info(f"Stored {results.articles_extracted} articles in database")
        
        # Note: In a real implementation, you would verify the articles
        # were stored correctly by querying the database
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Database integration example failed: {e}")
        logger.info("Note: This example requires a PostgreSQL database")


async def single_url_crawling_example():
    """
    Example: Crawling a single URL with detailed analysis.
    """
    logger.info("=== Single URL Crawling Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
    try:
        # Create enhanced crawler for single URL
        config = {
            'enable_ml_analysis': True,
            'enable_monitoring': True,
            'parser_config': {
                'enable_html_parsing': True,
                'enable_article_extraction': True,
                'enable_metadata_extraction': True,
                'enable_ml_analysis': True
            }
        }
        
        crawler = EnhancedNewsCrawler(config)
        
        # Crawl a single URL
        test_url = "https://httpbin.org/html"
        logger.info(f"Crawling single URL: {test_url}")
        
        article = await crawler.crawl_url(test_url)
        
        if article:
            logger.info("Article extracted successfully:")
            logger.info(f"  Title: {article.title}")
            logger.info(f"  Content length: {len(article.content)} characters")
            logger.info(f"  Quality score: {article.quality_score:.2f}")
            logger.info(f"  Source domain: {article.source_domain}")
            
            if article.ml_analysis:
                logger.info("ML Analysis results:")
                logger.info(f"  Sentiment: {article.ml_analysis.get('sentiment_label', 'N/A')}")
                logger.info(f"  Entities: {len(article.ml_analysis.get('entities', []))}")
        else:
            logger.warning("Failed to extract article")
        
        # Get crawler statistics
        stats = crawler.get_stats()
        logger.info(f"Crawler components available: {stats['availability']}")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Single URL crawling example failed: {e}")


async def performance_monitoring_example():
    """
    Example: Monitoring crawler performance and metrics.
    """
    logger.info("=== Performance Monitoring Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
    try:
        # Create crawler with monitoring enabled
        config = create_comprehensive_config(max_concurrent=5)
        config['enable_monitoring'] = True
        
        crawler = await create_comprehensive_crawler(config)
        
        # Add multiple targets for performance testing
        targets = []
        for i in range(5):
            targets.append(CrawlTarget(
                url=f"https://httpbin.org/delay/{i}",
                name=f"Delay Test {i}",
                priority=1,
                max_pages=1
            ))
        
        crawler.add_targets(targets)
        
        # Perform crawl with timing
        start_time = datetime.now()
        results = await crawler.crawl_all_targets()
        end_time = datetime.now()
        
        # Display performance metrics
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Performance Results:")
        logger.info(f"  Total duration: {duration:.2f} seconds")
        logger.info(f"  Targets processed: {results.targets_processed}")
        logger.info(f"  Average per target: {duration/max(results.targets_processed, 1):.2f} seconds")
        
        if results.performance_metrics:
            logger.info("Detailed performance metrics:")
            for metric, value in results.performance_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value}")
        
        # Get comprehensive stats
        stats = crawler.get_comprehensive_stats()
        logger.info(f"Session ID: {stats['session_id']}")
        logger.info(f"Configuration: {stats['configuration']}")
        
        await crawler.cleanup()
        
    except Exception as e:
        logger.error(f"Performance monitoring example failed: {e}")


async def save_results_example():
    """
    Example: Saving crawl results to files.
    """
    logger.info("=== Save Results Example ===")
    
    if not CRAWLER_AVAILABLE:
        logger.error("Crawler components not available")
        return
    
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
                'summary': article.summary,
                'source_domain': article.source_domain,
                'quality_score': article.quality_score,
                'extraction_metadata': article.extraction_metadata
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
            'success_rate': results.success_rate,
            'errors': results.errors
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
    """
    Main function to run all examples.
    """
    logger.info("Starting News Crawler Examples")
    logger.info("=" * 50)
    
    examples = [
        basic_crawling_example,
        single_url_crawling_example,
        stealth_crawling_example,
        performance_monitoring_example,
        save_results_example,
        # database_integration_example,  # Uncomment if you have a database setup
    ]
    
    for example in examples:
        try:
            await example()
            logger.info("Example completed successfully")
        except Exception as e:
            logger.error(f"Example failed: {e}")
        
        logger.info("-" * 30)
        await asyncio.sleep(1)  # Brief pause between examples
    
    logger.info("All examples completed")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())