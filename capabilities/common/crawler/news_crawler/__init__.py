"""
News Crawler Package
====================

Advanced news crawling system with comprehensive stealth capabilities,
content extraction, and integration with utils/config infrastructure.

Key Components:
- core/: Core crawler functionality (enhanced and deep crawling)
- bypass/: Cloudflare and anti-bot bypass mechanisms
- parsers/: Content parsing and extraction utilities
- config/: Unified configuration management using utils/config
- examples/: Usage examples and demonstrations

Features:
- **Enhanced Stealth Crawling**: Advanced browser mimicking and fingerprint spoofing
- **Deep Site Crawling**: Comprehensive site discovery and bulk article extraction
- **Bypass Systems**: Cloudflare, Captcha, and 403 error handling
- **Content Extraction**: Intelligent article and metadata extraction
- **Unified Configuration**: Uses utils/config infrastructure
- **Performance Monitoring**: Real-time performance tracking

Example Usage:
    Enhanced crawler with stealth:
    >>> from news_crawler.core import NewsCrawler
    >>> crawler = NewsCrawler()
    >>> article = await crawler.crawl_url("https://example.com/article")

    Deep crawling for entire sites:
    >>> from news_crawler.core import CloudScraperStealthCrawler
    >>> crawler = CloudScraperStealthCrawler()
    >>> articles = await crawler.crawl_entire_site("https://news-site.com")

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import logging

# Version information
__version__ = "4.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import core components
try:
    from .core.enhanced_news_crawler import (
        NewsCrawler,
        NewsArticle,
        StealthConfig,
        create_news_crawler,
        create_stealth_crawler
    )
    from .core.deep_crawling_news_crawler import (
        CloudScraperStealthCrawler,
        CrawlTarget,
        CrawlSession
    )
    from .core.improved_base_crawler import (
        ImprovedBaseCrawler,
        CrawlerConfig,
        CrawlRequest,
        CrawlResult,
        create_improved_crawler,
        create_news_optimized_crawler
    )
    CORE_AVAILABLE = True
    logger.debug("Core crawler components loaded")
except ImportError as e:
    logger.warning(f"Core crawler components not available: {e}")
    CORE_AVAILABLE = False
    NewsCrawler = None
    NewsArticle = None
    StealthConfig = None
    CloudScraperStealthCrawler = None
    CrawlTarget = None
    CrawlSession = None
    create_news_crawler = None
    create_stealth_crawler = None
    ImprovedBaseCrawler = None
    CrawlerConfig = None
    CrawlRequest = None
    CrawlResult = None
    create_improved_crawler = None
    create_news_optimized_crawler = None

# Import stealth components
try:
    from .stealth import (
        StealthEngine,
        StealthCrawler,
        UnifiedStealthOrchestrator,
        CloudScraperPriorityStealth,
        StealthConfig as StealthModuleConfig,
        create_stealth_crawler as create_stealth_module_crawler
    )
    STEALTH_AVAILABLE = True
    logger.debug("Stealth components loaded")
except ImportError as e:
    logger.warning(f"Stealth components not available: {e}")
    STEALTH_AVAILABLE = False
    StealthEngine = None
    StealthCrawler = None
    UnifiedStealthOrchestrator = None
    CloudScraperPriorityStealth = None
    StealthModuleConfig = None
    create_stealth_module_crawler = None


# Create StealthNewsCrawler class
class StealthNewsCrawler:
    """Stealth-enabled news crawler combining core and stealth functionality."""
    
    def __init__(self, stealth_config=None, crawler_config=None):
        if not STEALTH_AVAILABLE:
            raise ImportError("Stealth components not available")
        if not CORE_AVAILABLE:
            raise ImportError("Core crawler components not available")
            
        self.stealth_crawler = StealthCrawler(stealth_config)
        self.base_crawler = ImprovedBaseCrawler(crawler_config)
        
    async def crawl_url(self, url: str, **kwargs):
        """Crawl URL with stealth capabilities."""
        # Prepare stealth request
        stealth_config = await self.stealth_crawler.stealth_engine.prepare_request(url)
        
        # Create enhanced crawl request
        request = CrawlRequest(
            url=url,
            headers=stealth_config.get('headers', {}),
            **kwargs
        )
        
        # Execute crawl with base crawler
        result = await self.base_crawler.crawl_request(request)
        
        return result
    
    def get_stats(self):
        """Get combined statistics."""
        stealth_stats = self.stealth_crawler.stealth_engine.get_stats()
        crawler_stats = self.base_crawler.get_metrics()
        
        return {
            'stealth': stealth_stats,
            'crawler': crawler_stats
        }


def create_stealth_news_crawler(stealth_config=None, crawler_config=None):
    """Factory function to create StealthNewsCrawler."""
    return StealthNewsCrawler(stealth_config, crawler_config)


class BatchCrawler:
    """Batch crawler for processing multiple URLs efficiently."""
    
    def __init__(self, base_crawler=None, max_concurrent=10, batch_size=50):
        self.base_crawler = base_crawler or (ImprovedBaseCrawler() if CORE_AVAILABLE else None)
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.results = []
        self.failed_urls = []
        
    async def crawl_batch(self, urls, **kwargs):
        """Crawl a batch of URLs."""
        if not self.base_crawler:
            raise ImportError("Base crawler not available")
            
        results = []
        for i in range(0, len(urls), self.batch_size):
            batch = urls[i:i + self.batch_size]
            batch_results = await self.base_crawler.crawl_multiple(batch, **kwargs)
            results.extend(batch_results)
            
        return results
    
    def get_successful_results(self):
        """Get only successful crawl results."""
        return [r for r in self.results if r.status_code and r.status_code < 400]
    
    def get_failed_results(self):
        """Get failed crawl results."""
        return [r for r in self.results if not r.status_code or r.status_code >= 400]


def create_batch_crawler(max_concurrent=10, batch_size=50):
    """Factory function to create BatchCrawler."""
    return BatchCrawler(max_concurrent=max_concurrent, batch_size=batch_size)


# Import configuration system
try:
    from .config import (
        NewsConfigurationManager,
        NewsConfigurationAdapter,
        create_news_crawler_config,
        get_news_crawler_config_manager,
        get_crawler_config,
        UNIFIED_CONFIG_AVAILABLE
    )
    CONFIG_AVAILABLE = True
    logger.debug("Configuration system loaded")
except ImportError as e:
    logger.warning(f"Configuration system not available: {e}")
    CONFIG_AVAILABLE = False
    NewsConfigurationManager = None
    NewsConfigurationAdapter = None
    create_news_crawler_config = None
    get_news_crawler_config_manager = None
    get_crawler_config = None
    UNIFIED_CONFIG_AVAILABLE = False

# Import bypass and parsers
try:
    from .bypass import BypassManager, BypassConfig, create_bypass_manager
    BYPASS_AVAILABLE = True
except ImportError:
    BYPASS_AVAILABLE = False
    BypassManager = None
    BypassConfig = None
    create_bypass_manager = None

try:
    from .parsers import ContentParser, ParsedContent
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    ContentParser = None
    ParsedContent = None


# Convenience functions
def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    if CONFIG_AVAILABLE:
        try:
            return get_crawler_config()
        except:
            pass

    # Fallback configuration
    return {
        'max_concurrent_requests': 5,
        'requests_per_second': 2.0,
        'request_timeout': 30,
        'max_retries': 3,
        'enable_stealth': True,
        'enable_bypass': True,
        'enable_enhanced_stealth': True,
        'enable_ml_analysis': True
    }


async def crawl_url(url: str, config: Optional[Dict[str, Any]] = None) -> Optional[NewsArticle]:
    """
    Convenience function to crawl a single URL.

    Args:
        url: URL to crawl
        config: Optional configuration overrides

    Returns:
        NewsArticle if successful, None otherwise
    """
    if not CORE_AVAILABLE:
        raise ImportError("Core crawler components not available")

    effective_config = get_default_config()
    if config:
        effective_config.update(config)

    crawler = NewsCrawler(effective_config)
    try:
        return await crawler.crawl_url(url)
    finally:
        await crawler.cleanup()


async def crawl_multiple(urls: List[str], config: Optional[Dict[str, Any]] = None) -> List[NewsArticle]:
    """
    Convenience function to crawl multiple URLs.

    Args:
        urls: List of URLs to crawl
        config: Optional configuration overrides

    Returns:
        List of successfully crawled articles
    """
    if not CORE_AVAILABLE:
        raise ImportError("Core crawler components not available")

    effective_config = get_default_config()
    if config:
        effective_config.update(config)

    crawler = NewsCrawler(effective_config)
    articles = []

    try:
        for url in urls:
            try:
                article = await crawler.crawl_url(url)
                if article:
                    articles.append(article)
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")

        return articles
    finally:
        await crawler.cleanup()


def get_crawler_health() -> Dict[str, Any]:
    """Get health status of crawler components."""
    return {
        'version': __version__,
        'core_available': CORE_AVAILABLE,
        'config_available': CONFIG_AVAILABLE,
        'bypass_available': BYPASS_AVAILABLE,
        'parsers_available': PARSERS_AVAILABLE,
        'unified_config_available': UNIFIED_CONFIG_AVAILABLE,
        'status': 'healthy' if CORE_AVAILABLE else 'degraded'
    }


# Export all public components
__all__ = [
    # Core classes
    "NewsCrawler",
    "NewsArticle",
    "StealthConfig",
    "CloudScraperStealthCrawler",
    "CrawlTarget",
    "CrawlSession",
    "ImprovedBaseCrawler",
    "CrawlerConfig",
    "CrawlRequest",
    "CrawlResult",
    
    # Stealth classes
    "StealthNewsCrawler",
    "StealthEngine",
    "StealthCrawler",
    "UnifiedStealthOrchestrator",
    "CloudScraperPriorityStealth",
    
    # Batch processing
    "BatchCrawler",

    # Configuration
    "NewsConfigurationManager",
    "NewsConfigurationAdapter",
    "create_news_crawler_config",
    "get_news_crawler_config_manager",
    "get_crawler_config",

    # Bypass and parsers
    "BypassManager",
    "BypassConfig",
    "create_bypass_manager",
    "ContentParser",
    "ParsedContent",

    # Factory functions
    "create_news_crawler",
    "create_stealth_crawler",
    "create_improved_crawler",
    "create_news_optimized_crawler",
    "create_stealth_news_crawler",
    "create_batch_crawler",

    # Convenience functions
    "crawl_url",
    "crawl_multiple",
    "get_default_config",
    "get_crawler_health",

    # Availability flags
    "CORE_AVAILABLE",
    "CONFIG_AVAILABLE",
    "BYPASS_AVAILABLE",
    "PARSERS_AVAILABLE",
    "UNIFIED_CONFIG_AVAILABLE",

    # Version info
    "__version__",
    "__author__",
    "__license__"
]

# Module initialization
logger.info(f"News Crawler Package v{__version__} initialized")
logger.info(f"Core available: {CORE_AVAILABLE}, Config available: {CONFIG_AVAILABLE}")
logger.info(f"Bypass available: {BYPASS_AVAILABLE}, Parsers available: {PARSERS_AVAILABLE}")
