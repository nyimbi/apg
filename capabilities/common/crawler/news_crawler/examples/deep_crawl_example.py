"""
Deep Crawling News Crawler Example
==================================

Comprehensive example demonstrating deep site crawling with CloudScraper stealth,
targeting a minimum of 100 articles per site with real-time monitoring.

This example shows:
- Deep crawling of multiple news sites
- CloudScraper stealth integration
- Real-time progress monitoring
- Database storage
- Quality assessment
- Performance metrics

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Import the deep crawler
from lindela.packages_enhanced.crawlers.news_crawler import (
    CloudScraperStealthCrawler,
    CrawlTarget,
    create_deep_crawler,
    create_production_deep_crawler
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_crawl.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class DeepCrawlMonitor:
    """Real-time monitoring for deep crawling operations."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.sessions = []
    
    async def monitor_session(self, session, target_name: str):
        """Monitor a crawling session in real-time."""
        logger.info(f"Starting monitoring for {target_name}")
        
        while not session.end_time:
            # Log current progress
            logger.info(f"[{target_name}] Progress Update:")
            logger.info(f"  URLs discovered: {session.urls_discovered}")
            logger.info(f"  URLs processed: {session.urls_processed}")
            logger.info(f"  Articles extracted: {session.articles_extracted}")
            logger.info(f"  Target: {session.target.max_articles}")
            logger.info(f"  Success rate: {self._calculate_success_rate(session):.1f}%")
            logger.info(f"  Cloudflare bypasses: {session.cloudflare_bypasses}")
            
            # Check if target reached
            if session.articles_extracted >= session.target.max_articles:
                logger.info(f"[{target_name}] Target reached! ({session.articles_extracted} articles)")
                break
            
            await asyncio.sleep(10)  # Update every 10 seconds
    
    def _calculate_success_rate(self, session) -> float:
        """Calculate success rate for URL processing."""
        if session.urls_processed == 0:
            return 0.0
        return (session.articles_extracted / session.urls_processed) * 100
    
    def generate_report(self, sessions: List[Any]) -> Dict[str, Any]:
        """Generate comprehensive crawling report."""
        total_articles = sum(s.articles_extracted for s in sessions)
        total_urls = sum(s.urls_processed for s in sessions)
        total_duration = (datetime.now() - self.start_time).total_seconds()
        
        report = {
            "crawl_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": total_duration,
                "total_sites": len(sessions),
                "total_articles": total_articles,
                "total_urls_processed": total_urls,
                "overall_success_rate": (total_articles / max(total_urls, 1)) * 100
            },
            "site_results": []
        }
        
        for session in sessions:
            site_result = {
                "site_name": session.target.name,
                "site_url": session.target.url,
                "target_articles": session.target.max_articles,
                "articles_extracted": session.articles_extracted,
                "target_reached": session.target_reached,
                "urls_discovered": session.urls_discovered,
                "urls_processed": session.urls_processed,
                "success_rate": self._calculate_success_rate(session),
                "cloudflare_bypasses": session.cloudflare_bypasses,
                "user_agent_rotations": session.user_agent_rotations,
                "total_delay": session.delay_total,
                "duration": (session.end_time - session.start_time).total_seconds() if session.end_time else 0,
                "errors": len(session.errors)
            }
            report["site_results"].append(site_result)
        
        return report


async def example_single_site_deep_crawl():
    """Example: Deep crawl a single news site."""
    logger.info("=== Single Site Deep Crawl Example ===")
    
    # Create deep crawler with stealth
    crawler = await create_deep_crawler({
        'min_articles_per_site': 100,
        'max_depth': 5,
        'max_pages': 1000,
        'min_delay': 2.0,
        'max_delay': 5.0,
        'max_concurrent': 3,
        'debug': True
    })
    
    # Define crawl target
    target = CrawlTarget(
        url="https://www.bbc.com/news",
        name="BBC News",
        max_articles=120,
        max_depth=5,
        max_pages=1000,
        respect_robots_txt=True,
        include_patterns=[
            r'/news/',
            r'/world/',
            r'/politics/'
        ],
        exclude_patterns=[
            r'/sport/',
            r'/weather/',
            r'/iplayer/'
        ]
    )
    
    # Monitor crawling progress
    monitor = DeepCrawlMonitor()
    
    # Start crawling with monitoring
    logger.info(f"Starting deep crawl of {target.name}")
    session = await crawler.deep_crawl_site(target)
    
    # Log results
    logger.info("=== Crawl Results ===")
    logger.info(f"Site: {target.name}")
    logger.info(f"Articles extracted: {session.articles_extracted}")
    logger.info(f"Target reached: {session.target_reached}")
    logger.info(f"URLs processed: {session.urls_processed}")
    logger.info(f"Cloudflare bypasses: {session.cloudflare_bypasses}")
    logger.info(f"Duration: {session.end_time - session.start_time}")
    
    # Show sample articles
    logger.info("=== Sample Articles ===")
    for i, article in enumerate(session.articles[:5]):
        logger.info(f"Article {i+1}:")
        logger.info(f"  Title: {article.title[:100]}...")
        logger.info(f"  URL: {article.url}")
        logger.info(f"  Word count: {article.word_count}")
        logger.info(f"  Quality score: {article.quality_score:.2f}")
    
    return session


async def example_multiple_sites_crawl():
    """Example: Deep crawl multiple news sites simultaneously."""
    logger.info("=== Multiple Sites Deep Crawl Example ===")
    
    # Create production crawler
    crawler = await create_production_deep_crawler()
    
    # Define multiple targets
    targets = [
        CrawlTarget(
            url="https://www.reuters.com",
            name="Reuters",
            max_articles=100,
            priority=1,
            max_depth=4,
            include_patterns=[r'/world/', r'/business/', r'/politics/']
        ),
        CrawlTarget(
            url="https://www.theguardian.com",
            name="The Guardian", 
            max_articles=120,
            priority=1,
            max_depth=5,
            include_patterns=[r'/world/', r'/politics/', r'/international/']
        ),
        CrawlTarget(
            url="https://apnews.com",
            name="Associated Press",
            max_articles=80,
            priority=2,
            max_depth=4,
            include_patterns=[r'/article/', r'/hub/']
        )
    ]
    
    # Crawl all sites
    logger.info(f"Starting crawl of {len(targets)} sites")
    sessions = await crawler.crawl_multiple_sites(targets)
    
    # Generate comprehensive report
    monitor = DeepCrawlMonitor()
    report = monitor.generate_report(sessions)
    
    # Log summary
    logger.info("=== Multiple Sites Crawl Results ===")
    logger.info(f"Total sites: {report['crawl_summary']['total_sites']}")
    logger.info(f"Total articles: {report['crawl_summary']['total_articles']}")
    logger.info(f"Overall success rate: {report['crawl_summary']['overall_success_rate']:.1f}%")
    logger.info(f"Total duration: {report['crawl_summary']['duration_seconds']:.1f} seconds")
    
    # Log individual site results
    for site in report['site_results']:
        logger.info(f"\n{site['site_name']}:")
        logger.info(f"  Target: {site['target_articles']} articles")
        logger.info(f"  Extracted: {site['articles_extracted']} articles")
        logger.info(f"  Target reached: {site['target_reached']}")
        logger.info(f"  Success rate: {site['success_rate']:.1f}%")
        logger.info(f"  Cloudflare bypasses: {site['cloudflare_bypasses']}")
    
    # Save report to file
    report_file = f"crawl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Detailed report saved to: {report_file}")
    
    return sessions


async def example_custom_configuration():
    """Example: Custom crawler configuration for specific requirements."""
    logger.info("=== Custom Configuration Example ===")
    
    # Custom configuration for aggressive crawling
    custom_config = {
        'min_articles_per_site': 200,  # Higher target
        'max_depth': 7,                # Deeper crawling
        'max_pages': 2000,             # More pages
        'min_delay': 1.5,              # Faster requests (be careful!)
        'max_delay': 3.0,
        'max_concurrent': 2,           # Lower concurrency for stealth
        'timeout': 45,
        'enable_database': True,
        'enable_ml_analysis': True,
        'respect_robots_txt': True,
        'debug': False
    }
    
    # Create crawler with custom config
    crawler = CloudScraperStealthCrawler(custom_config)
    
    # Define target with custom selectors
    target = CrawlTarget(
        url="https://example-news.com",
        name="Example News Site",
        max_articles=200,
        max_depth=7,
        max_pages=2000,
        custom_selectors={
            'article': '.article-content',
            'title': 'h1.headline',
            'author': '.byline .author',
            'date': '.publish-date'
        },
        include_patterns=[
            r'/news/',
            r'/breaking/',
            r'/exclusive/'
        ],
        exclude_patterns=[
            r'/sports/',
            r'/entertainment/',
            r'/lifestyle/'
        ]
    )
    
    logger.info("Starting crawl with custom configuration")
    session = await crawler.deep_crawl_site(target)
    
    logger.info("=== Custom Configuration Results ===")
    logger.info(f"Articles extracted: {session.articles_extracted}")
    logger.info(f"Target reached: {session.target_reached}")
    logger.info(f"Average quality score: {sum(a.quality_score for a in session.articles) / len(session.articles):.2f}")
    
    return session


async def example_with_database_integration():
    """Example: Deep crawling with database storage."""
    logger.info("=== Database Integration Example ===")
    
    # Configuration with database enabled
    db_config = {
        'min_articles_per_site': 100,
        'enable_database': True,
        'database_url': 'postgresql://postgres:@localhost:5432/lindela',
        'min_delay': 2.0,
        'max_delay': 4.0,
        'max_concurrent': 3
    }
    
    # Create crawler with database integration
    crawler = CloudScraperStealthCrawler(db_config)
    
    # Define target
    target = CrawlTarget(
        url="https://www.aljazeera.com",
        name="Al Jazeera",
        max_articles=100,
        max_depth=4
    )
    
    logger.info("Starting crawl with database storage")
    session = await crawler.deep_crawl_site(target)
    
    logger.info("=== Database Integration Results ===")
    logger.info(f"Articles extracted: {session.articles_extracted}")
    logger.info(f"Articles stored in database: {session.articles_extracted}")
    logger.info("Check your PostgreSQL database for stored articles!")
    
    return session


async def example_stealth_monitoring():
    """Example: Monitoring stealth metrics during crawling."""
    logger.info("=== Stealth Monitoring Example ===")
    
    # Configuration with detailed stealth tracking
    stealth_config = {
        'min_articles_per_site': 50,  # Smaller target for demo
        'min_delay': 3.0,             # Conservative delays
        'max_delay': 7.0,
        'max_concurrent': 1,          # Very conservative
        'debug': True                 # Enable debug logging
    }
    
    crawler = CloudScraperStealthCrawler(stealth_config)
    
    target = CrawlTarget(
        url="https://www.cnn.com",
        name="CNN",
        max_articles=50,
        max_depth=3
    )
    
    logger.info("Starting crawl with stealth monitoring")
    session = await crawler.deep_crawl_site(target)
    
    logger.info("=== Stealth Metrics ===")
    logger.info(f"Cloudflare bypasses: {session.cloudflare_bypasses}")
    logger.info(f"User agent rotations: {session.user_agent_rotations}")
    logger.info(f"Total delay time: {session.delay_total:.1f} seconds")
    logger.info(f"Average delay per request: {session.delay_total / max(session.urls_processed, 1):.2f} seconds")
    
    return session


async def main():
    """Run all examples."""
    logger.info("Starting Deep Crawling News Crawler Examples")
    logger.info("=" * 60)
    
    try:
        # Example 1: Single site deep crawl
        await example_single_site_deep_crawl()
        await asyncio.sleep(5)  # Brief pause between examples
        
        # Example 2: Multiple sites crawl
        await example_multiple_sites_crawl()
        await asyncio.sleep(5)
        
        # Example 3: Custom configuration
        await example_custom_configuration()
        await asyncio.sleep(5)
        
        # Example 4: Database integration
        await example_with_database_integration()
        await asyncio.sleep(5)
        
        # Example 5: Stealth monitoring
        await example_stealth_monitoring()
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        raise
    
    logger.info("All examples completed successfully!")


if __name__ == "__main__":
    # Check dependencies
    try:
        import cloudscraper
        logger.info("✓ CloudScraper available")
    except ImportError:
        logger.error("✗ CloudScraper not available. Install with: pip install cloudscraper")
        exit(1)
    
    try:
        import newspaper
        import trafilatura
        logger.info("✓ Content extraction libraries available")
    except ImportError:
        logger.warning("⚠ Content extraction libraries not fully available")
        logger.warning("Install with: pip install newspaper3k trafilatura")
    
    # Run examples
    asyncio.run(main())