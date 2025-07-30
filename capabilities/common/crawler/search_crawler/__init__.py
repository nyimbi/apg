"""
Search Crawler Package
======================

Multi-search engine crawler for conflict monitoring in Horn of Africa.
Searches using multiple engines, scrapes results, and ranks by relevance.

Features:
- Multiple search engines (Google, Bing, DuckDuckGo, Yandex)
- Horn of Africa conflict keyword detection
- Intelligent result ranking and ordering
- Content scraping and analysis
- Stealth techniques for search engines
- Database integration

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any
import logging

# Version information
__version__ = "1.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import core components with fallback handling
try:
    from .core.search_crawler import SearchCrawler, SearchCrawlerConfig
    from .core.conflict_search_crawler import ConflictSearchCrawler
    _CORE_AVAILABLE = True
    logger.debug("Search crawler core components loaded successfully")
except ImportError as e:
    logger.warning(f"Core search crawler components not available: {e}")
    _CORE_AVAILABLE = False
    
    # Placeholder classes
    class SearchCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Search crawler components not available")
    
    class ConflictSearchCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Search crawler components not available")

# Import search content pipeline components
try:
    from .search_content_pipeline import (
        SearchContentPipeline,
        SearchContentPipelineConfig,
        EnrichedSearchResult,
        create_search_content_pipeline,
        create_conflict_search_content_pipeline,
        create_horn_africa_search_pipeline
    )
    _SEARCH_CONTENT_PIPELINE_AVAILABLE = True
    logger.debug("Search content pipeline components loaded successfully")
except ImportError as e:
    logger.warning(f"Search content pipeline components not available: {e}")
    _SEARCH_CONTENT_PIPELINE_AVAILABLE = False
    
    # Placeholder classes
    class SearchContentPipeline:
        def __init__(self, *args, **kwargs):
            raise ImportError("Search content pipeline components not available")
    
    class SearchContentPipelineConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("Search content pipeline components not available")
    
    class EnrichedSearchResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("Search content pipeline components not available")

# Import Crawlee-enhanced components
try:
    from .core.crawlee_enhanced_search_crawler import (
        CrawleeEnhancedSearchCrawler,
        CrawleeSearchConfig,
        CrawleeEnhancedResult,
        create_crawlee_search_config,
        create_crawlee_search_crawler,
        CRAWLEE_AVAILABLE
    )
    _CRAWLEE_ENHANCED_AVAILABLE = True
    logger.debug("Crawlee-enhanced search crawler components loaded successfully")
except ImportError as e:
    logger.warning(f"Crawlee-enhanced components not available: {e}")
    _CRAWLEE_ENHANCED_AVAILABLE = False
    CRAWLEE_AVAILABLE = False
    
    # Placeholder classes
    class CrawleeEnhancedSearchCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Crawlee-enhanced search crawler not available")
    
    class CrawleeSearchConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("Crawlee-enhanced search crawler not available")
    
    class CrawleeEnhancedResult:
        def __init__(self, *args, **kwargs):
            raise ImportError("Crawlee-enhanced search crawler not available")

# Import search engines
try:
    from .engines import (
        GoogleSearchEngine, BingSearchEngine, DuckDuckGoSearchEngine,
        YandexSearchEngine, BaiduSearchEngine, YahooSearchEngine,
        StartpageSearchEngine, SearXSearchEngine, BraveSearchEngine,
        MojeekSearchEngine, SwisscowsSearchEngine, SEARCH_ENGINES
    )
    _ENGINES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Search engines not available: {e}")
    _ENGINES_AVAILABLE = False

# Import keywords and analysis
try:
    from .keywords.conflict_keywords import ConflictKeywordManager
    from .keywords.horn_of_africa_keywords import HornOfAfricaKeywords
    from .keywords.keyword_analyzer import KeywordAnalyzer
    _ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Analysis components not available: {e}")
    _ANALYSIS_AVAILABLE = False

# Import configuration
try:
    from .core.search_crawler import SearchCrawlerConfig
    from .core.conflict_search_crawler import ConflictSearchConfig
    _CONFIG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Configuration not available: {e}")
    _CONFIG_AVAILABLE = False


def create_conflict_search_crawler(
    config: Optional[Dict[str, Any]] = None,
    enable_stealth: bool = True,
    max_results_per_engine: int = 50
) -> 'ConflictSearchCrawler':
    """
    Create a conflict monitoring search crawler.
    
    Args:
        config: Optional configuration dictionary
        enable_stealth: Enable stealth techniques for search engines
        max_results_per_engine: Maximum results per search engine
        
    Returns:
        Configured ConflictSearchCrawler instance
    """
    if not _CORE_AVAILABLE:
        raise ImportError("Search crawler core components not available")
    
    if config is None and _CONFIG_AVAILABLE:
        config = ConflictSearchConfig(
            use_stealth=enable_stealth,
            max_results_per_engine=max_results_per_engine
        )
    elif config is None:
        config = ConflictSearchConfig()
    elif isinstance(config, dict) and _CONFIG_AVAILABLE:
        # Convert dictionary to ConflictSearchConfig object
        config = ConflictSearchConfig(**config)
    elif isinstance(config, dict):
        # Fallback to basic ConflictSearchConfig if config module not available
        config = ConflictSearchConfig()
    
    return ConflictSearchCrawler(config=config)


async def create_crawlee_enhanced_search_crawler(
    config: Optional[Dict[str, Any]] = None,
    engines: Optional[List[str]] = None,
    enable_content_extraction: bool = True,
    target_countries: Optional[List[str]] = None
) -> 'CrawleeEnhancedSearchCrawler':
    """
    Create a Crawlee-enhanced search crawler with content downloading.
    
    Args:
        config: Optional configuration dictionary
        engines: List of search engines to use
        enable_content_extraction: Enable full content extraction with Crawlee
        target_countries: Target countries for geographic relevance
        
    Returns:
        Configured CrawleeEnhancedSearchCrawler instance
    """
    if not _CRAWLEE_ENHANCED_AVAILABLE:
        raise ImportError("Crawlee-enhanced search crawler components not available")
    
    if config is None:
        crawler_config = create_crawlee_search_config(
            engines=engines,
            enable_content_extraction=enable_content_extraction,
            target_countries=target_countries
        )
    elif isinstance(config, dict):
        crawler_config = CrawleeSearchConfig(**config)
    else:
        crawler_config = config
    
    return await create_crawlee_search_crawler(crawler_config)


def create_general_search_crawler(
    engines: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> 'SearchCrawler':
    """
    Create a general-purpose search crawler.
    
    Args:
        engines: List of search engines to use
        config: Optional configuration dictionary
        
    Returns:
        Configured SearchCrawler instance
    """
    if not _CORE_AVAILABLE:
        raise ImportError("Search crawler core components not available")
    
    if config is None:
        config = SearchCrawlerConfig()
        if engines:
            config.engines = engines
    elif isinstance(config, dict):
        crawler_config = SearchCrawlerConfig(**config)
        if engines:
            crawler_config.engines = engines
        config = crawler_config
    
    return SearchCrawler(config=config)


def create_search_content_pipeline_factory(
    search_engines: Optional[List[str]] = None,
    database_url: str = "postgresql:///lnd",
    enable_content_download: bool = True,
    enable_database_storage: bool = True,
    **config_kwargs
) -> 'SearchContentPipeline':
    """
    Factory function to create a complete search content pipeline.
    
    Args:
        search_engines: List of search engines to use
        database_url: Database connection string
        enable_content_download: Enable full content download via gen_crawler
        enable_database_storage: Enable database storage
        **config_kwargs: Additional configuration parameters
        
    Returns:
        SearchContentPipeline: Configured pipeline
    """
    if not _SEARCH_CONTENT_PIPELINE_AVAILABLE:
        raise ImportError("Search content pipeline components not available")
    
    return create_search_content_pipeline(
        search_engines=search_engines,
        database_url=database_url,
        enable_content_download=enable_content_download,
        enable_database_storage=enable_database_storage,
        **config_kwargs
    )

def create_conflict_monitoring_pipeline(
    database_url: str = "postgresql:///lnd"
) -> 'SearchContentPipeline':
    """
    Create search content pipeline optimized for conflict monitoring.
    
    Args:
        database_url: Database connection string
        
    Returns:
        SearchContentPipeline: Conflict monitoring optimized pipeline
    """
    if not _SEARCH_CONTENT_PIPELINE_AVAILABLE:
        raise ImportError("Search content pipeline components not available")
    
    return create_conflict_search_content_pipeline(database_url)

def create_regional_monitoring_pipeline(
    database_url: str = "postgresql:///lnd"
) -> 'SearchContentPipeline':
    """
    Create search pipeline optimized for Horn of Africa content discovery.
    
    Args:
        database_url: Database connection string
        
    Returns:
        SearchContentPipeline: Horn of Africa optimized pipeline
    """
    if not _SEARCH_CONTENT_PIPELINE_AVAILABLE:
        raise ImportError("Search content pipeline components not available")
    
    return create_horn_africa_search_pipeline(database_url)

def get_search_crawler_health() -> Dict[str, Any]:
    """Get health status of search crawler components."""
    all_available = all([
        _CORE_AVAILABLE, 
        _ENGINES_AVAILABLE, 
        _ANALYSIS_AVAILABLE, 
        _CRAWLEE_ENHANCED_AVAILABLE,
        _SEARCH_CONTENT_PIPELINE_AVAILABLE
    ])
    
    return {
        'status': 'healthy' if all_available else 'degraded',
        'core_available': _CORE_AVAILABLE,
        'engines_available': _ENGINES_AVAILABLE,
        'analysis_available': _ANALYSIS_AVAILABLE,
        'config_available': _CONFIG_AVAILABLE,
        'crawlee_enhanced_available': _CRAWLEE_ENHANCED_AVAILABLE,
        'crawlee_library_available': CRAWLEE_AVAILABLE,
        'search_content_pipeline_available': _SEARCH_CONTENT_PIPELINE_AVAILABLE,
        'version': __version__
    }


# CLI availability check
try:
    from .cli import SearchCrawlerCLI, CLIConfig, CrawlerMode, OutputFormat
    _CLI_AVAILABLE = True
except ImportError:
    _CLI_AVAILABLE = False
    SearchCrawlerCLI = None
    CLIConfig = None
    CrawlerMode = None
    OutputFormat = None

# Export public components
__all__ = [
    # Core classes
    'SearchCrawler',
    'ConflictSearchCrawler',
    
    # Search Content Pipeline classes
    'SearchContentPipeline',
    'SearchContentPipelineConfig',
    'EnrichedSearchResult',
    
    # Crawlee-enhanced classes
    'CrawleeEnhancedSearchCrawler',
    'CrawleeSearchConfig',
    'CrawleeEnhancedResult',
    
    # Factory functions
    'create_conflict_search_crawler',
    'create_general_search_crawler',
    'create_crawlee_enhanced_search_crawler',
    'create_crawlee_search_config',
    'create_crawlee_search_crawler',
    
    # Search Content Pipeline factory functions
    'create_search_content_pipeline',
    'create_conflict_search_content_pipeline',
    'create_horn_africa_search_pipeline',
    'create_search_content_pipeline_factory',
    'create_conflict_monitoring_pipeline',
    'create_regional_monitoring_pipeline',
    
    # Search engines (all 11 engines)
    'GoogleSearchEngine',
    'BingSearchEngine', 
    'DuckDuckGoSearchEngine',
    'YandexSearchEngine',
    'BaiduSearchEngine',
    'YahooSearchEngine',
    'StartpageSearchEngine',
    'SearXSearchEngine',
    'BraveSearchEngine',
    'MojeekSearchEngine',
    'SwisscowsSearchEngine',
    'SEARCH_ENGINES',
    
    # Keywords and analysis
    'ConflictKeywordManager',
    'HornOfAfricaKeywords',
    'KeywordAnalyzer',
    
    # Configuration
    'SearchCrawlerConfig',
    'ConflictSearchConfig',
    
    # CLI components (if available)
    'SearchCrawlerCLI',
    'CLIConfig',
    'CrawlerMode',
    'OutputFormat',
    
    # Utilities
    'get_search_crawler_health',
    
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Availability flags
    'CRAWLEE_AVAILABLE',
    '_CLI_AVAILABLE',
    '_SEARCH_CONTENT_PIPELINE_AVAILABLE'
]

# Module initialization
logger.info(f"Search Crawler Package v{__version__} initialized")
logger.info(f"Components available - Core: {_CORE_AVAILABLE}, Engines: {_ENGINES_AVAILABLE}, Analysis: {_ANALYSIS_AVAILABLE}, Crawlee: {_CRAWLEE_ENHANCED_AVAILABLE}, SearchContentPipeline: {_SEARCH_CONTENT_PIPELINE_AVAILABLE}, CLI: {_CLI_AVAILABLE}")