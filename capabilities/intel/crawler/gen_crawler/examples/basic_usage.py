"""
Basic Usage Example for Gen Crawler
===================================

This example demonstrates basic usage of the gen_crawler package
for full-site crawling using AdaptivePlaywrightCrawler.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def basic_site_crawl():
    """Demonstrate basic site crawling."""
    try:
        # Import gen_crawler components
        from gen_crawler.core import GenCrawler, create_gen_crawler
        
        logger.info("🚀 Starting basic site crawl example")
        
        # Create crawler with default configuration
        crawler = create_gen_crawler()
        
        # Initialize the crawler
        await crawler.initialize()
        
        # Crawl a test site
        test_url = "https://example.com"
        logger.info(f"Crawling site: {test_url}")
        
        result = await crawler.crawl_site(test_url)
        
        # Display results
        logger.info("📊 Crawl Results:")
        logger.info(f"   Total pages: {result.total_pages}")
        logger.info(f"   Successful pages: {result.successful_pages}")
        logger.info(f"   Failed pages: {result.failed_pages}")
        logger.info(f"   Success rate: {result.success_rate:.1f}%")
        logger.info(f"   Total time: {result.total_time:.1f}s")
        
        # Show sample content
        if result.pages:
            sample_page = result.pages[0]
            logger.info(f"   Sample page: {sample_page.url}")
            logger.info(f"   Sample title: {sample_page.title[:100]}...")
            logger.info(f"   Sample content type: {sample_page.content_type}")
            logger.info(f"   Sample word count: {sample_page.word_count}")
        
        # Cleanup
        await crawler.cleanup()
        
        return result
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure Crawlee is installed: pip install 'crawlee[all]'")
        return None
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        return None

async def advanced_site_crawl():
    """Demonstrate advanced site crawling with custom configuration."""
    try:
        from gen_crawler.core import GenCrawler
        from gen_crawler.config import create_gen_config
        
        logger.info("🚀 Starting advanced site crawl example")
        
        # Create custom configuration
        config = create_gen_config()
        config.settings.performance.max_pages_per_site = 50
        config.settings.performance.max_concurrent = 3
        config.settings.performance.crawl_delay = 1.0
        config.settings.adaptive.enable_adaptive_crawling = True
        
        # Get crawler config
        crawler_config = config.get_crawler_config()
        
        # Create crawler with custom config
        crawler = GenCrawler(crawler_config)
        await crawler.initialize()
        
        # Crawl multiple sites
        test_sites = [
            "https://example.com",
            "https://httpbin.org"
        ]
        
        all_results = []
        
        for site_url in test_sites:
            logger.info(f"Crawling site: {site_url}")
            
            try:
                result = await crawler.crawl_site(site_url)
                all_results.append(result)
                
                logger.info(f"✅ {site_url}: {result.total_pages} pages, {result.success_rate:.1f}% success")
                
            except Exception as e:
                logger.error(f"❌ Failed to crawl {site_url}: {e}")
        
        # Display combined statistics
        total_pages = sum(r.total_pages for r in all_results)
        total_successful = sum(r.successful_pages for r in all_results)
        overall_success_rate = (total_successful / total_pages * 100) if total_pages > 0 else 0
        
        logger.info("📊 Combined Results:")
        logger.info(f"   Sites crawled: {len(all_results)}")
        logger.info(f"   Total pages: {total_pages}")
        logger.info(f"   Total successful: {total_successful}")
        logger.info(f"   Overall success rate: {overall_success_rate:.1f}%")
        
        # Cleanup
        await crawler.cleanup()
        
        return all_results
        
    except Exception as e:
        logger.error(f"Advanced crawl error: {e}")
        return []

async def analyze_crawler_performance():
    """Demonstrate crawler performance analysis."""
    try:
        from gen_crawler.core import GenCrawler, AdaptiveCrawler
        
        logger.info("🚀 Starting crawler performance analysis")
        
        # Create crawler and adaptive manager
        crawler = GenCrawler()
        await crawler.initialize()
        
        adaptive_crawler = AdaptiveCrawler()
        
        # Test site
        test_url = "https://httpbin.org"
        
        # Crawl and analyze
        result = await crawler.crawl_site(test_url)
        
        # Get performance statistics
        crawler_stats = crawler.get_statistics()
        adaptive_stats = adaptive_crawler.get_crawler_stats()
        
        logger.info("📊 Performance Analysis:")
        logger.info(f"   Pages crawled: {crawler_stats['pages_crawled']}")
        logger.info(f"   Links discovered: {crawler_stats['links_discovered']}")
        logger.info(f"   Errors: {crawler_stats['errors']}")
        
        # Site profile analysis
        site_profile = adaptive_crawler.get_site_profile(test_url)
        logger.info(f"   Site success rate: {site_profile.success_rate:.1f}%")
        logger.info(f"   Average load time: {site_profile.average_load_time:.2f}s")
        logger.info(f"   Performance score: {site_profile.performance_score:.2f}")
        
        # Cleanup
        await crawler.cleanup()
        
        return {
            'crawl_result': result,
            'crawler_stats': crawler_stats,
            'adaptive_stats': adaptive_stats
        }
        
    except Exception as e:
        logger.error(f"Performance analysis error: {e}")
        return None

async def test_content_analysis():
    """Demonstrate content analysis capabilities."""
    try:
        from gen_crawler.parsers import GenContentParser, create_content_parser
        
        logger.info("🚀 Starting content analysis example")
        
        # Create content parser
        parser = create_content_parser()
        
        # Sample HTML content
        sample_html = """
        <html>
        <head>
            <title>Sample News Article</title>
            <meta name="author" content="John Doe">
            <meta name="description" content="This is a sample article">
        </head>
        <body>
            <article>
                <h1>Breaking News: Technology Advances</h1>
                <p>This is the first paragraph of a sample news article about technology advances.</p>
                <p>This is the second paragraph with more detailed information about the topic.</p>
                <p>The article continues with additional analysis and expert opinions.</p>
            </article>
        </body>
        </html>
        """
        
        # Parse content
        parsed_content = parser.parse_content("https://example.com/article", sample_html)
        
        # Display analysis results
        logger.info("📊 Content Analysis Results:")
        logger.info(f"   Title: {parsed_content.title}")
        logger.info(f"   Content type: {parsed_content.content_type}")
        logger.info(f"   Word count: {parsed_content.word_count}")
        logger.info(f"   Quality score: {parsed_content.quality_score:.2f}")
        logger.info(f"   Is article: {parsed_content.is_article}")
        logger.info(f"   Is high quality: {parsed_content.is_high_quality}")
        logger.info(f"   Authors: {parsed_content.authors}")
        logger.info(f"   Extraction method: {parsed_content.extraction_method}")
        
        # Show parser status
        parser_status = parser.get_parser_status()
        logger.info(f"   Available methods: {parser_status['available_methods']}")
        
        return parsed_content
        
    except Exception as e:
        logger.error(f"Content analysis error: {e}")
        return None

async def main():
    """Run all examples."""
    logger.info("🎯 Gen Crawler Examples")
    logger.info("=" * 50)
    
    # Check package health
    try:
        from gen_crawler import get_gen_crawler_health
        health = get_gen_crawler_health()
        logger.info(f"Package status: {health['status']}")
        logger.info(f"Available capabilities: {health['capabilities']}")
        
        if health['status'] != 'healthy':
            logger.warning("Gen crawler not fully functional. Some examples may fail.")
    except ImportError:
        logger.error("Gen crawler package not available")
        return
    
    examples = [
        ("Basic Site Crawl", basic_site_crawl),
        ("Advanced Site Crawl", advanced_site_crawl),
        ("Performance Analysis", analyze_crawler_performance),
        ("Content Analysis", test_content_analysis)
    ]
    
    for name, example_func in examples:
        logger.info(f"\n📋 Running: {name}")
        logger.info("-" * 30)
        
        try:
            result = await example_func()
            if result:
                logger.info(f"✅ {name} completed successfully")
            else:
                logger.warning(f"⚠️ {name} completed with issues")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    logger.info("\n🎉 All examples completed!")

if __name__ == "__main__":
    asyncio.run(main())