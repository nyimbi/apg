"""
Unified Configuration Adapter for NewsAPI Crawler
=================================================

Adapter that bridges the NewsAPI crawler's legacy configuration system
with the new unified configuration system from packages_enhanced/utils/config.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import Optional, Dict, Any

# Import the new unified configuration system
try:
    from ...utils.config import (
        UnifiedCrawlerConfiguration,
        APICredentialsConfiguration,
        StealthConfiguration,
        CrawlerPerformanceConfiguration,
        ContentExtractionConfiguration,
        CrawlerCacheConfiguration,
        CrawlerType,
        StealthLevel,
        get_global_config_manager,
        initialize_global_config_manager
    )
    UNIFIED_CONFIG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Unified configuration system not available: {e}")
    UNIFIED_CONFIG_AVAILABLE = False
    UnifiedCrawlerConfiguration = None

# Import legacy configuration for compatibility
try:
    from .config.configuration import NewsAPIConfig as LegacyNewsAPIConfig
    LEGACY_CONFIG_AVAILABLE = True
except ImportError:
    LEGACY_CONFIG_AVAILABLE = False
    LegacyNewsAPIConfig = None

logger = logging.getLogger(__name__)


class NewsAPIConfigurationManager:
    """Configuration manager for the NewsAPI crawler."""
    
    def __init__(self, use_unified: bool = True):
        self.use_unified = use_unified and UNIFIED_CONFIG_AVAILABLE
        self.unified_config: Optional[UnifiedCrawlerConfiguration] = None
        self.legacy_config: Optional[Any] = None
        self._config_manager = None
        
        if self.use_unified:
            self._config_manager = get_global_config_manager()
    
    async def initialize(self):
        """Initialize the configuration manager."""
        if self.use_unified and self._config_manager:
            try:
                await initialize_global_config_manager()
                logger.info("Unified configuration manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize unified config manager: {e}")
                self.use_unified = False
    
    def _create_default_unified_config(self) -> UnifiedCrawlerConfiguration:
        """Create default unified configuration for NewsAPI crawler."""
        # Create API credentials for NewsAPI
        api_credentials = APICredentialsConfiguration(
            newsapi_key=None,  # Should be set via environment
            newsapi_base_url="https://newsapi.org/v2",
            newsapi_daily_limit=100
        )
        
        # Create minimal stealth configuration (APIs don't need stealth)
        stealth = StealthConfiguration(
            level=StealthLevel.BASIC,
            rotate_user_agents=False
        )
        
        # Create performance configuration for API calls
        performance = CrawlerPerformanceConfiguration(
            max_concurrent_requests=5,
            requests_per_second=0.5,  # Conservative for free tier
            request_timeout=30,
            max_retries=3,
            batch_size=100
        )
        
        # Create content extraction configuration
        content_extraction = ContentExtractionConfiguration(
            extract_text=True,
            extract_metadata=True,
            extract_title=True,
            extract_author=True,
            extract_publish_date=True,
            min_content_length=50
        )
        
        # Create cache configuration for API responses
        cache = CrawlerCacheConfiguration(
            enabled=True,
            response_ttl=3600,  # 1 hour for API responses
            content_ttl=7200    # 2 hours for article content
        )
        
        return UnifiedCrawlerConfiguration(
            crawler_type=CrawlerType.NEWS,
            crawler_name="newsapi_crawler",
            api_credentials=api_credentials,
            stealth=stealth,
            performance=performance,
            content_extraction=content_extraction,
            cache=cache
        )


# Export classes
__all__ = [
    'NewsAPIConfigurationManager',
    'UNIFIED_CONFIG_AVAILABLE',
    'LEGACY_CONFIG_AVAILABLE'
]