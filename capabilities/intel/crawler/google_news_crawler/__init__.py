"""
Google News Crawler Package
============================

Enterprise-grade Google News crawler with filtering, stealth capabilities,
and integration with existing systems.

This package provides:
- Google News client with multi-source aggregation
- Site filtering and credibility assessment
- Stealth crawling capabilities
- PostgreSQL integration
- Backward compatibility with original GNews API

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__email__ = "nyimbi@datacraft.co.ke"
__company__ = "Datacraft"

# Import main classes for easy access
try:
    from .api.google_news_client import (
        # Main client classes
        GoogleNewsClient,
        GNewsCompatibilityWrapper,

        # Configuration and utility classes
        NewsSource,
        SiteFilteringEngine,
        NewsSourceConfig,

        # Data classes and enums
        SourceType,
        ContentFormat,
        CrawlPriority,
        GeographicalFocus,
        SourceCredibilityMetrics,

        # Factory functions
        create_gnews_client,
        create_basic_gnews_client,
        create_crawlee_gnews_client,
        create_sample_configuration,
        load_source_configuration,
    )
    
    # Import enhanced client components
    from .enhanced_client import (
        EnhancedGoogleNewsClient,
        EnhancedGoogleNewsConfig,
        create_enhanced_google_news_client,
        create_horn_africa_news_client
    )
    
    # Import database components
    from .database import (
        InformationUnitsManager,
        GoogleNewsRecord,
        create_information_units_manager,
        TokenBucketRateLimiter,
        RateLimitConfig,
        create_rate_limiter
    )
    
    # Import resilience components
    from .resilience import (
        CircuitBreaker,
        ErrorHandler,
        create_circuit_breaker,
        create_google_news_circuit_breaker,
        create_error_handler
    )
    
    # Import integration layer
    from .integration import (
        IntegratedGoogleNewsClient,
        IntegrationConfig,
        create_integrated_client,
        create_production_ready_client,
        create_basic_client,
        migrate_to_enhanced_features
    )
    
    # Import content pipeline (Google News + Gen Crawler integration)
    from .content_pipeline import (
        GoogleNewsContentPipeline,
        ContentPipelineConfig,
        EnrichedNewsItem,
        create_content_pipeline,
        create_horn_africa_content_pipeline
    )

    # Import configuration classes
    from .config import CrawlerConfig, ConfigurationManager, get_config, load_config

    # Import parser classes
    from .parsers import (
        BaseParser, ParseResult, ArticleData, ParseStatus, ContentType,
        ParserRegistry, parser_registry
    )

    # Import Crawlee integration (if available)
    try:
        from .crawlee_integration import (
            CrawleeConfig, ArticleResult, CrawleeNewsEnhancer,
            create_crawlee_config, create_crawlee_enhancer, CRAWLEE_AVAILABLE
        )
        CRAWLEE_INTEGRATION_AVAILABLE = True
    except ImportError:
        CRAWLEE_INTEGRATION_AVAILABLE = False
        CrawleeConfig = None
        ArticleResult = None
        CrawleeNewsEnhancer = None
        create_crawlee_config = None
        create_crawlee_enhancer = None
        CRAWLEE_AVAILABLE = False

    # Define what gets imported with "from gnews_crawler import *"
    __all__ = [
        # Main classes
        'GoogleNewsClient',
        'GNewsCompatibilityWrapper',
        'NewsSource',
        'SiteFilteringEngine',
        'NewsSourceConfig',

        # Enhanced client classes
        'EnhancedGoogleNewsClient',
        'EnhancedGoogleNewsConfig',

        # Database components
        'InformationUnitsManager',
        'GoogleNewsRecord',
        'TokenBucketRateLimiter',
        'RateLimitConfig',

        # Resilience components
        'CircuitBreaker',
        'ErrorHandler',
        
        # Integration layer
        'IntegratedGoogleNewsClient',
        'IntegrationConfig',
        
        # Content Pipeline (Google News + Gen Crawler)
        'GoogleNewsContentPipeline',
        'ContentPipelineConfig',
        'EnrichedNewsItem',

        # Enums and data classes
        'SourceType',
        'ContentFormat',
        'CrawlPriority',
        'GeographicalFocus',
        'SourceCredibilityMetrics',

        # Factory functions
        'create_gnews_client',
        'create_basic_gnews_client',
        'create_crawlee_gnews_client',
        'create_sample_configuration',
        'load_source_configuration',
        
        # Enhanced factory functions
        'create_enhanced_google_news_client',
        'create_horn_africa_news_client',
        'create_information_units_manager',
        'create_rate_limiter',
        'create_circuit_breaker',
        'create_google_news_circuit_breaker',
        'create_error_handler',
        
        # Integration factory functions
        'create_integrated_client',
        'create_production_ready_client',
        'create_basic_client',
        'migrate_to_enhanced_features',
        
        # Content pipeline factory functions
        'create_content_pipeline',
        'create_horn_africa_content_pipeline',

        # Configuration
        'CrawlerConfig',
        'ConfigurationManager',
        'get_config',
        'load_config',

        # Parsers
        'BaseParser',
        'ParseResult',
        'ArticleData',
        'ParseStatus',
        'ContentType',
        'ParserRegistry',
        'parser_registry',

        # Crawlee integration
        'CrawleeConfig',
        'ArticleResult',
        'CrawleeNewsEnhancer',
        'create_crawlee_config',
        'create_crawlee_enhancer',
        'CRAWLEE_AVAILABLE',
        'CRAWLEE_INTEGRATION_AVAILABLE',

        # CLI interface (if available)
        'cli_available',
    ]
    
    # Check if CLI is available
    try:
        from .cli.main import main_cli
        cli_available = True
        __all__.extend(['main_cli'])
    except ImportError:
        cli_available = False

except ImportError as e:
    # If imports fail, provide a helpful error message
    import warnings
    warnings.warn(
        f"Could not import core components: {e}\n"
        "Please ensure all dependencies are installed:\n"
        "pip install aiohttp asyncpg feedparser beautifulsoup4",
        ImportWarning
    )

    # Define empty __all__ to prevent import errors
    __all__ = []

# Add aliases for backward compatibility
try:
    GoogleNewsCrawler = GoogleNewsClient
    GoogleNewsResult = ArticleData  # Use ArticleData as result type
    
    aliases_to_add = ['GoogleNewsCrawler', 'GoogleNewsResult']
    for alias in aliases_to_add:
        if alias not in __all__:
            __all__.append(alias)
except NameError:
    # If main components are not available, create minimal placeholders
    from dataclasses import dataclass
    from typing import Optional, Dict, Any
    
    @dataclass
    class GoogleNewsResult:
        """Placeholder result class when main components are not available."""
        title: str = ""
        url: str = ""
        content: str = ""
        published_date: Optional[str] = None
        source: str = ""
        metadata: Dict[str, Any] = None
        
        def __post_init__(self):
            if self.metadata is None:
                self.metadata = {}
    
    class GoogleNewsCrawler:
        """Placeholder GoogleNewsCrawler when main components are not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("GoogleNewsCrawler dependencies not available. Install with: pip install aiohttp asyncpg feedparser beautifulsoup4")
    
    class GoogleNewsClient:
        """Placeholder GoogleNewsClient when main components are not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("GoogleNewsClient dependencies not available. Install with: pip install aiohttp asyncpg feedparser beautifulsoup4")
    
    __all__.extend(['GoogleNewsCrawler', 'GoogleNewsClient', 'GoogleNewsResult'])

# Add factory function for backward compatibility
def create_google_news_crawler(*args, **kwargs):
    """Factory function to create GoogleNewsCrawler."""
    return GoogleNewsCrawler(*args, **kwargs)

# Add to exports
__all__.append('create_google_news_crawler')

# Package metadata
PACKAGE_INFO = {
    'name': 'Enhanced Google News Crawler',
    'version': __version__,
    'description': 'Enterprise-grade Google News crawler with advanced filtering and stealth capabilities',
    'author': __author__,
    'company': __company__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'keywords': ['news', 'crawler', 'google-news', 'rss', 'scraping', 'journalism'],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: General',
    ]
}

def get_version():
    """Get the package version."""
    return __version__

def get_package_info():
    """Get complete package information."""
    return PACKAGE_INFO.copy()

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    optional_deps = []

    # Core dependencies
    core_deps = [
        ('aiohttp', 'HTTP client for async operations'),
        ('asyncpg', 'PostgreSQL async driver'),
        ('feedparser', 'RSS/Atom feed parsing'),
    ]

    # Optional dependencies
    opt_deps = [
        ('beautifulsoup4', 'HTML parsing'),
        ('pandas', 'Data manipulation'),
        ('textblob', 'Text analysis and sentiment'),
        ('scikit-learn', 'Machine learning features'),
        ('pydantic', 'Data validation'),
        ('PyYAML', 'Configuration file support'),
    ]

    for dep_name, description in core_deps:
        try:
            __import__(dep_name.replace('-', '_'))
        except ImportError:
            missing_deps.append((dep_name, description))

    for dep_name, description in opt_deps:
        try:
            __import__(dep_name.replace('-', '_'))
        except ImportError:
            optional_deps.append((dep_name, description))

    return {
        'missing_required': missing_deps,
        'missing_optional': optional_deps,
        'all_available': len(missing_deps) == 0
    }

# Convenience function for quick setup
def quick_setup_guide():
    """Print a quick setup guide for the package."""
    print(f"""
Enhanced Google News Crawler v{__version__}
==========================================

Quick Setup Guide:
1. Install dependencies: pip install -r requirements.txt
2. Set up PostgreSQL database
3. Configure your database connection
4. Initialize the crawler:

   from gnews_crawler import create_enhanced_gnews_client

   client = await create_enhanced_gnews_client(
       db_manager=your_db_manager,
       config=your_config
   )

For detailed documentation, see the README.md file.
For examples, check the test files and example scripts.

Dependencies Status:
""")

    deps = check_dependencies()
    if deps['all_available']:
        print("✅ All required dependencies are available!")
    else:
        print("❌ Missing required dependencies:")
        for dep, desc in deps['missing_required']:
            print(f"   - {dep}: {desc}")

    if deps['missing_optional']:
        print("\n⚠️  Missing optional dependencies (some features may be limited):")
        for dep, desc in deps['missing_optional']:
            print(f"   - {dep}: {desc}")
