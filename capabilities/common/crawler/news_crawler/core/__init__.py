"""
News Crawler Core Module
========================

Core crawling functionality and base classes for the news crawler system.
Integrates with packages_enhanced utilities for enhanced functionality.

Components:
- BaseCrawler: Base crawler class with utils integration
- NewsCrawler: Main news crawling implementation
- EnhancedNewsCrawler: Advanced crawler with ML integration
- CrawlerFactory: Factory for creating configured crawlers
- CrawlerConfig: Configuration management

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any
import logging

# Version information
__version__ = "4.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import core components with fallback handling
try:
    from .enhanced_news_crawler import NewsCrawler, CrawlSession, NewsArticle, create_news_crawler, create_stealth_crawler
    from .deep_crawling_news_crawler import CloudScraperStealthCrawler, CrawlTarget, create_deep_crawler, create_production_deep_crawler
    
    # Create aliases for backward compatibility
    BaseCrawler = NewsCrawler
    EnhancedNewsCrawler = NewsCrawler  # Backward compatibility
    
    # Create CrawlerConfig placeholder
    class CrawlerConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # Create CrawlerResult placeholder
    class CrawlerResult:
        def __init__(self, url: str, success: bool = False, content: str = "", **kwargs):
            self.url = url
            self.success = success
            self.content = content
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    _CORE_COMPONENTS_AVAILABLE = True
    logger.debug("Core crawler components loaded successfully")
except ImportError as e:
    logger.warning(f"Some core components not available: {e}")
    _CORE_COMPONENTS_AVAILABLE = False
    
    # Placeholder classes
    class BaseCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core crawler components not available")
    
    class NewsCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core crawler components not available")
    
    class EnhancedNewsCrawler:
        def __init__(self, *args, **kwargs):
            raise ImportError("Core crawler components not available")


class CrawlerFactory:
    """Factory for creating configured crawler instances."""
    
    @staticmethod
    def create_crawler(
        crawler_type: str = "enhanced",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'BaseCrawler':
        """
        Create a crawler instance based on type and configuration.
        
        Args:
            crawler_type: Type of crawler ('basic', 'enhanced', 'stealth')
            config: Optional configuration dictionary
            **kwargs: Additional configuration parameters
            
        Returns:
            Configured crawler instance
        """
        if not _CORE_COMPONENTS_AVAILABLE:
            raise ImportError("Core crawler components not available")
        
        config = config or {}
        config.update(kwargs)
        
        if crawler_type == "basic":
            return NewsCrawler(config=config)
        elif crawler_type == "enhanced":
            return EnhancedNewsCrawler(config=config)
        else:
            raise ValueError(f"Unknown crawler type: {crawler_type}")


# Utility functions
def get_default_config() -> Dict[str, Any]:
    """Get default crawler configuration."""
    return {
        'max_concurrent': 5,
        'timeout': 30,
        'retries': 3,
        'delay_range': (1, 3),
        'enable_stealth': False,
        'enable_ml_scoring': True,
        'enable_monitoring': True
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate crawler configuration."""
    required_fields = ['max_concurrent', 'timeout', 'retries']
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    if config['max_concurrent'] <= 0:
        raise ValueError("max_concurrent must be positive")
    
    if config['timeout'] <= 0:
        raise ValueError("timeout must be positive")
    
    return True


def get_core_health() -> Dict[str, Any]:
    """Get health status of core components."""
    return {
        'status': 'healthy' if _CORE_COMPONENTS_AVAILABLE else 'degraded',
        'components_available': _CORE_COMPONENTS_AVAILABLE,
        'version': __version__,
        'supported_crawlers': ['basic', 'enhanced'] if _CORE_COMPONENTS_AVAILABLE else []
    }


# Export all public components
__all__ = [
    # Core classes
    'NewsCrawler',
    'BaseCrawler',  # Alias for backward compatibility
    'EnhancedNewsCrawler',  # Alias for backward compatibility
    'CrawlerFactory',
    'CrawlerConfig',
    'CrawlerResult',
    'NewsArticle',
    'CrawlSession',
    
    # Factory functions
    'create_news_crawler',
    'create_stealth_crawler',
    
    # Utility functions
    'get_default_config',
    'validate_config',
    'get_core_health',
    
    # Version info
    '__version__',
    '__author__',
    '__license__'
]

# Module initialization
logger.info(f"News Crawler Core Module v{__version__} initialized")
logger.info(f"Core components available: {_CORE_COMPONENTS_AVAILABLE}")