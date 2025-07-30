"""
Generation Crawler Package (gen_crawler)
========================================

Next-generation web crawler built on Crawlee's AdaptivePlaywrightCrawler
for comprehensive full-site crawling with modern asyncio architecture.

Key Features:
- **AdaptivePlaywrightCrawler**: Advanced Crawlee-based crawling engine
- **Full Site Crawling**: Comprehensive site discovery and content extraction
- **Intelligent Adaptation**: Automatically switches between HTTP and browser-based crawling
- **Modern Architecture**: Built on asyncio and latest Python patterns
- **Performance Optimized**: Concurrent crawling with intelligent resource management
- **Content-Aware**: AI-powered content analysis and classification
- **Database Integration**: Seamless integration with Lindela database systems

Components:
- core/: Core crawler implementations using AdaptivePlaywrightCrawler
- config/: Configuration management and unified config integration
- parsers/: Advanced content parsing and extraction utilities  
- examples/: Usage examples and demonstration scripts
- docs/: Documentation and architectural guides

Use Cases:
- Full news site crawling and archiving
- Content discovery and monitoring
- Competitive intelligence gathering
- Research data collection
- News feed aggregation

Example Usage:
    Basic full-site crawling:
    >>> from gen_crawler.core import GenCrawler
    >>> crawler = GenCrawler()
    >>> results = await crawler.crawl_site("https://news-site.com")

    Advanced crawling with configuration:
    >>> from gen_crawler.core import GenCrawler
    >>> from gen_crawler.config import GenCrawlerConfig
    >>> config = GenCrawlerConfig(max_pages=500, enable_ai_analysis=True)
    >>> crawler = GenCrawler(config)
    >>> results = await crawler.crawl_site("https://example.com")

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Union, Any
import logging

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Check Crawlee availability
try:
    from crawlee.crawlers import AdaptivePlaywrightCrawler, AdaptivePlaywrightCrawlingContext
    from crawlee import Request
    CRAWLEE_AVAILABLE = True
    logger.info("Crawlee components available")
except ImportError as e:
    CRAWLEE_AVAILABLE = False
    AdaptivePlaywrightCrawler = None
    AdaptivePlaywrightCrawlingContext = None
    Request = None
    logger.warning(f"Crawlee not available: {e}")

# Import core components
try:
    from .core.gen_crawler import (
        GenCrawler,
        GenCrawlResult,
        GenSiteResult,
        create_gen_crawler,
        create_gen_crawler_with_database
    )
    from .core.adaptive_crawler import (
        AdaptiveCrawler,
        CrawlStrategy,
        SiteProfile
    )
    CORE_AVAILABLE = True
    logger.debug("Core crawler components loaded")
except ImportError as e:
    logger.warning(f"Core crawler components not available: {e}")
    CORE_AVAILABLE = False
    GenCrawler = None
    GenCrawlResult = None
    GenSiteResult = None
    create_gen_crawler = None
    create_gen_crawler_with_database = None
    AdaptiveCrawler = None
    CrawlStrategy = None
    SiteProfile = None

# Import configuration system
try:
    from .config.gen_config import (
        GenCrawlerConfig,
        GenCrawlerSettings,
        create_gen_config,
        get_default_gen_config
    )
    CONFIG_AVAILABLE = True
    logger.debug("Configuration system loaded")
except ImportError as e:
    logger.warning(f"Configuration system not available: {e}")
    CONFIG_AVAILABLE = False
    GenCrawlerConfig = None
    GenCrawlerSettings = None
    create_gen_config = None
    get_default_gen_config = None

# Import parsers
try:
    from .parsers.content_parser import (
        GenContentParser,
        ParsedSiteContent,
        ContentAnalyzer
    )
    PARSERS_AVAILABLE = True
    logger.debug("Parser components loaded")
except ImportError as e:
    logger.warning(f"Parser components not available: {e}")
    PARSERS_AVAILABLE = False
    GenContentParser = None
    ParsedSiteContent = None
    ContentAnalyzer = None

# Convenience functions
async def crawl_site(url: str, config: Optional[Dict[str, Any]] = None) -> Optional['GenSiteResult']:
    """
    Convenience function to crawl an entire site.
    
    Args:
        url: Base URL of the site to crawl
        config: Optional configuration overrides
        
    Returns:
        GenSiteResult if successful, None otherwise
    """
    if not CORE_AVAILABLE or not CRAWLEE_AVAILABLE:
        raise ImportError("Core crawler components or Crawlee not available")
    
    effective_config = get_default_gen_config() if CONFIG_AVAILABLE else {}
    if config:
        effective_config.update(config)
        
    crawler = GenCrawler(effective_config)
    
    try:
        return await crawler.crawl_site(url)
    finally:
        await crawler.cleanup()

async def crawl_multiple_sites(urls: List[str], config: Optional[Dict[str, Any]] = None) -> List['GenSiteResult']:
    """
    Convenience function to crawl multiple sites.
    
    Args:
        urls: List of base URLs to crawl
        config: Optional configuration overrides
        
    Returns:
        List of GenSiteResult objects
    """
    if not CORE_AVAILABLE or not CRAWLEE_AVAILABLE:
        raise ImportError("Core crawler components or Crawlee not available")
    
    effective_config = get_default_gen_config() if CONFIG_AVAILABLE else {}
    if config:
        effective_config.update(config)
        
    crawler = GenCrawler(effective_config)
    results = []
    
    try:
        for url in urls:
            try:
                result = await crawler.crawl_site(url)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Failed to crawl site {url}: {e}")
        
        return results
    finally:
        await crawler.cleanup()

def get_gen_crawler_health() -> Dict[str, Any]:
    """Get health status of gen_crawler components."""
    return {
        'version': __version__,
        'crawlee_available': CRAWLEE_AVAILABLE,
        'core_available': CORE_AVAILABLE,
        'config_available': CONFIG_AVAILABLE,
        'parsers_available': PARSERS_AVAILABLE,
        'status': 'healthy' if (CRAWLEE_AVAILABLE and CORE_AVAILABLE) else 'degraded',
        'capabilities': {
            'full_site_crawling': CRAWLEE_AVAILABLE and CORE_AVAILABLE,
            'adaptive_crawling': CRAWLEE_AVAILABLE and CORE_AVAILABLE,
            'content_analysis': PARSERS_AVAILABLE,
            'configuration_management': CONFIG_AVAILABLE
        }
    }

# Export all public components
__all__ = [
    # Core classes
    "GenCrawler",
    "GenCrawlResult", 
    "GenSiteResult",
    "AdaptiveCrawler",
    "CrawlStrategy",
    "SiteProfile",
    
    # Configuration
    "GenCrawlerConfig",
    "GenCrawlerSettings",
    "create_gen_config",
    "get_default_gen_config",
    
    # Parsers
    "GenContentParser",
    "ParsedSiteContent", 
    "ContentAnalyzer",
    
    # Factory functions
    "create_gen_crawler",
    "create_gen_crawler_with_database",
    
    # Convenience functions
    "crawl_site",
    "crawl_multiple_sites",
    "get_gen_crawler_health",
    
    # Availability flags
    "CRAWLEE_AVAILABLE",
    "CORE_AVAILABLE",
    "CONFIG_AVAILABLE", 
    "PARSERS_AVAILABLE",
    
    # Version info
    "__version__",
    "__author__",
    "__license__"
]

# Module initialization
logger.info(f"Gen Crawler Package v{__version__} initialized")
logger.info(f"Crawlee available: {CRAWLEE_AVAILABLE}, Core available: {CORE_AVAILABLE}")
logger.info(f"Config available: {CONFIG_AVAILABLE}, Parsers available: {PARSERS_AVAILABLE}")

if not CRAWLEE_AVAILABLE:
    logger.warning("Crawlee not available. Install with: pip install 'crawlee[all]'")
if not (CRAWLEE_AVAILABLE and CORE_AVAILABLE):
    logger.warning("Gen crawler not fully functional. Some features may be unavailable.")