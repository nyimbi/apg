"""
Unified Configuration Adapter for News Crawler
==============================================

Adapter that bridges the news crawler's legacy configuration system
with the new unified configuration system from packages_enhanced/utils/config.
Provides backward compatibility while enabling new configuration features.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

# Import the new unified configuration system
try:
    from ....utils.config import (
        UnifiedCrawlerConfiguration,
        NewsSpecificConfiguration,
        APICredentialsConfiguration,
        StealthConfiguration,
        BypassConfiguration,
        CrawlerPerformanceConfiguration,
        ContentExtractionConfiguration,
        CrawlerCacheConfiguration,
        DatabaseConfiguration,
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

# Legacy configuration no longer needed - all using unified config
LEGACY_CONFIG_AVAILABLE = False
LegacyCrawlerConfiguration = None
LegacyDatabaseConfig = None
LegacyPerformanceConfig = None
LegacyStealthConfig = None
LegacyBypassConfig = None
LegacyMLConfig = None

logger = logging.getLogger(__name__)


class NewsConfigurationAdapter:
    """
    Adapter that converts between legacy and unified configuration formats
    for the news crawler system.
    """
    
    @staticmethod
    def legacy_to_unified(legacy_config: Any) -> Optional[UnifiedCrawlerConfiguration]:
        """
        Convert legacy news crawler configuration to unified format.
        
        Args:
            legacy_config: Legacy CrawlerConfiguration object
            
        Returns:
            UnifiedCrawlerConfiguration object or None if conversion fails
        """
        if not UNIFIED_CONFIG_AVAILABLE or not legacy_config:
            return None
        
        try:
            # Create API credentials configuration
            api_credentials = APICredentialsConfiguration()
            
            # Create stealth configuration
            stealth = StealthConfiguration()
            if hasattr(legacy_config, 'stealth') and legacy_config.stealth:
                stealth.level = StealthLevel.MODERATE if legacy_config.stealth.enable_stealth else StealthLevel.NONE
                stealth.rotate_user_agents = legacy_config.stealth.randomize_user_agent
                stealth.min_request_delay = getattr(legacy_config.stealth, 'min_delay', 1.0)
                stealth.max_request_delay = getattr(legacy_config.stealth, 'max_delay', 3.0)
                stealth.use_proxies = getattr(legacy_config.stealth, 'use_proxy_rotation', False)
            
            # Create bypass configuration
            bypass = BypassConfiguration()
            if hasattr(legacy_config, 'bypass') and legacy_config.bypass:
                bypass.cloudflare_enabled = getattr(legacy_config.bypass, 'cloudflare_bypass', False)
                bypass.captcha_enabled = getattr(legacy_config.bypass, 'captcha_solving', False)
                bypass.browser_automation = getattr(legacy_config.bypass, 'browser_automation', False)
            
            # Create performance configuration
            performance = CrawlerPerformanceConfiguration()
            if hasattr(legacy_config, 'performance') and legacy_config.performance:
                performance.max_concurrent_requests = legacy_config.performance.max_concurrent_requests
                performance.requests_per_second = legacy_config.performance.requests_per_second
                performance.request_timeout = legacy_config.performance.request_timeout
                performance.max_retries = legacy_config.performance.max_retries
                performance.batch_size = legacy_config.performance.batch_size
            
            # Create content extraction configuration
            content_extraction = ContentExtractionConfiguration()
            # Use defaults for content extraction as legacy config doesn't have these details
            
            # Create cache configuration
            cache = CrawlerCacheConfiguration()
            # Use defaults for cache as legacy config uses simple boolean
            
            # Create news-specific configuration
            news_specific = NewsSpecificConfiguration()
            if hasattr(legacy_config, 'sources'):
                news_specific.news_sites = getattr(legacy_config.sources, 'news_sites', [])
                news_specific.rss_feeds = getattr(legacy_config.sources, 'rss_feeds', [])
            
            # Create unified configuration
            unified_config = UnifiedCrawlerConfiguration(
                crawler_type=CrawlerType.NEWS,
                crawler_name="news_crawler",
                api_credentials=api_credentials,
                stealth=stealth,
                bypass=bypass,
                performance=performance,
                content_extraction=content_extraction,
                cache=cache
            )
            
            logger.info("Successfully converted legacy configuration to unified format")
            return unified_config
            
        except Exception as e:
            logger.error(f"Failed to convert legacy configuration: {e}")
            return None
    
    @staticmethod
    def unified_to_legacy(unified_config: UnifiedCrawlerConfiguration) -> Optional[Dict[str, Any]]:
        """
        Convert unified configuration to dictionary format for backward compatibility.
        
        Args:
            unified_config: UnifiedCrawlerConfiguration object
            
        Returns:
            Dictionary configuration or None if conversion fails
        """
        if not unified_config:
            return None
        
        try:
            # Convert to simple dictionary format
            legacy_dict = {
                'max_concurrent_requests': unified_config.performance.max_concurrent_requests,
                'requests_per_second': unified_config.performance.requests_per_second,
                'request_timeout': unified_config.performance.request_timeout,
                'max_retries': unified_config.performance.max_retries,
                'batch_size': unified_config.performance.batch_size,
                'enable_stealth': (unified_config.stealth.level != StealthLevel.NONE),
                'randomize_user_agent': unified_config.stealth.rotate_user_agents,
                'use_proxies': unified_config.stealth.use_proxies,
                'min_request_delay': unified_config.stealth.min_request_delay,
                'max_request_delay': unified_config.stealth.max_request_delay,
                'cloudflare_enabled': unified_config.bypass.cloudflare_enabled,
                'captcha_enabled': unified_config.bypass.captcha_enabled,
                'browser_automation': unified_config.bypass.browser_automation,
                'cache_enabled': unified_config.cache.enabled,
                'save_parsed_content': unified_config.save_parsed_content,
                'output_format': unified_config.output_format
            }
            
            logger.info("Successfully converted unified configuration to dictionary format")
            return legacy_dict
            
        except Exception as e:
            logger.error(f"Failed to convert unified configuration: {e}")
            return None


class NewsConfigurationManager:
    """
    Configuration manager for the news crawler that handles both 
    legacy and unified configuration systems.
    """
    
    def __init__(self, use_unified: bool = True):
        self.use_unified = use_unified and UNIFIED_CONFIG_AVAILABLE
        self.adapter = NewsConfigurationAdapter()
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
    
    async def load_configuration(self, 
                               config_file: Optional[str] = None,
                               context: str = "news_crawler") -> Any:
        """
        Load configuration from file or create default.
        
        Args:
            config_file: Path to configuration file (optional)
            context: Configuration context name
            
        Returns:
            Configuration object (unified or legacy)
        """
        if self.use_unified:
            try:
                # Try to load unified configuration
                if config_file:
                    self.unified_config = await self._config_manager.load_configuration(
                        context, "unified_crawler", UnifiedCrawlerConfiguration, config_file
                    )
                else:
                    # Create default news crawler configuration
                    self.unified_config = self._create_default_unified_config()
                    
                logger.info("Loaded unified news crawler configuration")
                return self.unified_config
                
            except Exception as e:
                logger.error(f"Failed to load unified configuration: {e}")
                # Fall back to legacy configuration
                self.use_unified = False
        
        # Legacy configuration not available - return minimal default
        logger.warning("No configuration system available, using minimal defaults")
        return self._create_minimal_config()
    
    def _create_default_unified_config(self) -> UnifiedCrawlerConfiguration:
        """Create default unified configuration for news crawler."""
        # Create optimized news crawler configuration
        stealth = StealthConfiguration(
            level=StealthLevel.MODERATE,
            rotate_user_agents=True,
            min_request_delay=1.0,
            max_request_delay=3.0,
            randomize_headers=True
        )
        
        performance = CrawlerPerformanceConfiguration(
            max_concurrent_requests=5,  # Conservative for news sites
            requests_per_second=2.0,    # Respectful rate limiting
            request_timeout=30,
            max_retries=3,
            batch_size=50
        )
        
        content_extraction = ContentExtractionConfiguration(
            extract_text=True,
            extract_links=True,
            extract_metadata=True,
            extract_title=True,
            extract_author=True,
            extract_publish_date=True,
            min_content_length=200,  # Minimum for news articles
            language_detection=True
        )
        
        cache = CrawlerCacheConfiguration(
            enabled=True,
            response_ttl=3600,          # 1 hour for responses
            content_ttl=86400,          # 24 hours for content
        )
        
        return UnifiedCrawlerConfiguration(
            crawler_type=CrawlerType.NEWS,
            crawler_name="enhanced_news_crawler",
            stealth=stealth,
            performance=performance,
            content_extraction=content_extraction,
            cache=cache,
            output_format="json",
            save_parsed_content=True
        )
    
    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal configuration as fallback."""
        return {
            "crawler_type": "news",
            "max_concurrent_requests": 5,
            "requests_per_second": 2.0,
            "request_timeout": 30,
            "enable_stealth": True,
            "cache_responses": True
        }
    
    def get_configuration(self) -> Any:
        """Get the current configuration object."""
        return self.unified_config if self.use_unified else self.legacy_config
    
    def get_database_config(self) -> Optional['DatabaseConfiguration']:
        """Get database configuration from current config."""
        if self.use_unified and self.unified_config:
            # Get database config from global manager
            if self._config_manager:
                try:
                    return self._config_manager.get_configuration_typed(
                        "news_crawler", "database", DatabaseConfiguration
                    )
                except Exception as e:
                    logger.warning(f"Failed to get typed database configuration: {e}")
                    # Fall back to creating a default configuration
                    return DatabaseConfiguration(
                        host='localhost',
                        port=5432,
                        database='lindela_news',
                        username='crawler_user',
                        password=''
                    )
        # No legacy database config support - return default
        elif self.legacy_config and isinstance(self.legacy_config, dict):
            # Return basic database config if available in dict
            return DatabaseConfiguration(
                host=self.legacy_config.get('db_host', 'localhost'),
                port=self.legacy_config.get('db_port', 5432),
                database=self.legacy_config.get('db_name', 'news_crawler'),
                username=self.legacy_config.get('db_username', 'crawler_user'),
                password=self.legacy_config.get('db_password', '')
            )
        
        return None
    
    def is_using_unified_config(self) -> bool:
        """Check if using unified configuration system."""
        return self.use_unified


# Convenience functions for backward compatibility
async def create_news_crawler_config(config_file: Optional[str] = None,
                                    use_unified: bool = True) -> Any:
    """
    Create news crawler configuration.
    
    Args:
        config_file: Path to configuration file
        use_unified: Whether to use unified configuration system
        
    Returns:
        Configuration object
    """
    manager = NewsConfigurationManager(use_unified=use_unified)
    await manager.initialize()
    return await manager.load_configuration(config_file)


def get_news_crawler_config_manager(use_unified: bool = True) -> NewsConfigurationManager:
    """
    Get news crawler configuration manager.
    
    Args:
        use_unified: Whether to use unified configuration system
        
    Returns:
        NewsConfigurationManager instance
    """
    return NewsConfigurationManager(use_unified=use_unified)


# Export all public classes and functions
__all__ = [
    'NewsConfigurationAdapter',
    'NewsConfigurationManager',
    'create_news_crawler_config',
    'get_news_crawler_config_manager',
    'UNIFIED_CONFIG_AVAILABLE',
    'LEGACY_CONFIG_AVAILABLE'
]