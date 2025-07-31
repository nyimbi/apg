"""
Unified Configuration Adapter for YouTube Crawler
=================================================

Adapter that bridges the YouTube crawler's legacy configuration system
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
    from ...utils.config import (
        UnifiedCrawlerConfiguration,
        YouTubeSpecificConfiguration,
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

# Import legacy configuration for compatibility
try:
    from .config import (
        CrawlerConfig as LegacyCrawlerConfig,
        APIConfig as LegacyAPIConfig,
        ScrapingConfig as LegacyScrapingConfig,
        DatabaseConfig as LegacyDatabaseConfig,
        CacheConfig as LegacyCacheConfig,
        FilteringConfig as LegacyFilteringConfig,
        ExtractionConfig as LegacyExtractionConfig,
        PerformanceConfig as LegacyPerformanceConfig,
        SecurityConfig as LegacySecurityConfig,
        MonitoringConfig as LegacyMonitoringConfig,
        CrawlMode,
        GeographicalFocus
    )
    LEGACY_CONFIG_AVAILABLE = True
except ImportError:
    LEGACY_CONFIG_AVAILABLE = False
    LegacyCrawlerConfig = None

logger = logging.getLogger(__name__)


class YouTubeConfigurationAdapter:
    """
    Adapter that converts between legacy and unified configuration formats
    for the YouTube crawler system.
    """
    
    @staticmethod
    def legacy_to_unified(legacy_config: Any) -> Optional[UnifiedCrawlerConfiguration]:
        """
        Convert legacy YouTube crawler configuration to unified format.
        
        Args:
            legacy_config: Legacy CrawlerConfig object
            
        Returns:
            UnifiedCrawlerConfiguration object or None if conversion fails
        """
        if not UNIFIED_CONFIG_AVAILABLE or not legacy_config:
            return None
        
        try:
            # Create API credentials configuration
            api_credentials = APICredentialsConfiguration()
            if hasattr(legacy_config, 'api') and legacy_config.api:
                api_credentials.youtube_api_key = legacy_config.api.api_key
                api_credentials.youtube_quota_limit = legacy_config.api.quota_limit
                api_credentials.youtube_requests_per_minute = legacy_config.api.requests_per_minute
            
            # Create stealth configuration
            stealth = StealthConfiguration()
            if hasattr(legacy_config, 'scraping') and legacy_config.scraping:
                stealth.level = StealthLevel.MODERATE if legacy_config.scraping.enable_stealth else StealthLevel.NONE
                stealth.user_agent_list = legacy_config.scraping.user_agents
                stealth.min_request_delay = legacy_config.scraping.request_delay
                stealth.max_request_delay = legacy_config.scraping.request_delay * 2
                stealth.random_delay = legacy_config.scraping.random_delay
                stealth.proxy_list = legacy_config.scraping.proxy_list
                stealth.use_proxies = legacy_config.scraping.proxy_rotation
            
            # Create bypass configuration
            bypass = BypassConfiguration()
            # YouTube typically doesn't need heavy bypass mechanisms
            
            # Create performance configuration
            performance = CrawlerPerformanceConfiguration()
            if hasattr(legacy_config, 'performance') and legacy_config.performance:
                performance.max_concurrent_requests = legacy_config.performance.max_concurrent_requests
                performance.requests_per_second = getattr(legacy_config.performance, 'requests_per_second', 2.0)
                performance.request_timeout = getattr(legacy_config.performance, 'timeout', 30)
                performance.max_retries = getattr(legacy_config.performance, 'max_retries', 3)
                performance.batch_size = getattr(legacy_config.performance, 'batch_size', 50)
            
            # Create content extraction configuration
            content_extraction = ContentExtractionConfiguration()
            if hasattr(legacy_config, 'extraction') and legacy_config.extraction:
                content_extraction.extract_text = legacy_config.extraction.extract_transcripts
                content_extraction.extract_metadata = True
                content_extraction.extract_title = True
                content_extraction.extract_links = True
                content_extraction.download_images = legacy_config.extraction.extract_thumbnails
            
            # Create cache configuration
            cache = CrawlerCacheConfiguration()
            if hasattr(legacy_config, 'cache') and legacy_config.cache:
                cache.enabled = legacy_config.cache.enabled
                cache.response_ttl = getattr(legacy_config.cache, 'ttl', 3600)
                cache.max_cache_size_mb = getattr(legacy_config.cache, 'max_size_mb', 512)
            
            # Create YouTube-specific configuration
            youtube_specific = YouTubeSpecificConfiguration()
            if hasattr(legacy_config, 'filtering') and legacy_config.filtering:
                youtube_specific.min_duration_seconds = legacy_config.filtering.min_duration
                youtube_specific.max_duration_seconds = legacy_config.filtering.max_duration
                youtube_specific.min_view_count = legacy_config.filtering.min_views
                youtube_specific.extract_transcripts = getattr(legacy_config.filtering, 'extract_transcripts', True)
                youtube_specific.extract_comments = getattr(legacy_config.filtering, 'extract_comments', False)
                youtube_specific.extract_thumbnails = getattr(legacy_config.filtering, 'extract_thumbnails', False)
            
            # Create unified configuration
            unified_config = UnifiedCrawlerConfiguration(
                crawler_type=CrawlerType.YOUTUBE,
                crawler_name="youtube_crawler",
                api_credentials=api_credentials,
                stealth=stealth,
                bypass=bypass,
                performance=performance,
                content_extraction=content_extraction,
                cache=cache
            )
            
            logger.info("Successfully converted legacy YouTube configuration to unified format")
            return unified_config
            
        except Exception as e:
            logger.error(f"Failed to convert legacy YouTube configuration: {e}")
            return None
    
    @staticmethod
    def unified_to_legacy(unified_config: UnifiedCrawlerConfiguration) -> Optional[Any]:
        """
        Convert unified configuration to legacy format for backward compatibility.
        
        Args:
            unified_config: UnifiedCrawlerConfiguration object
            
        Returns:
            Legacy configuration object or None if conversion fails
        """
        if not LEGACY_CONFIG_AVAILABLE or not unified_config:
            return None
        
        try:
            # Create legacy API config
            api_config = LegacyAPIConfig(
                api_key=unified_config.api_credentials.youtube_api_key,
                quota_limit=unified_config.api_credentials.youtube_quota_limit,
                requests_per_minute=unified_config.api_credentials.youtube_requests_per_minute,
                enable_quota_monitoring=True,
                fallback_to_scraping=True
            )
            
            # Create legacy scraping config
            scraping_config = LegacyScrapingConfig(
                enable_stealth=(unified_config.stealth.level != StealthLevel.NONE),
                user_agents=unified_config.stealth.user_agent_list,
                request_delay=unified_config.stealth.min_request_delay,
                random_delay=unified_config.stealth.random_delay,
                max_retries=unified_config.performance.max_retries,
                timeout=unified_config.performance.request_timeout,
                proxy_rotation=unified_config.stealth.use_proxies,
                proxy_list=unified_config.stealth.proxy_list
            )
            
            # Create legacy database config
            database_config = LegacyDatabaseConfig()
            
            # Create legacy cache config
            cache_config = LegacyCacheConfig(
                enabled=unified_config.cache.enabled,
                ttl=unified_config.cache.response_ttl,
                max_size_mb=unified_config.cache.max_cache_size_mb
            )
            
            # Create legacy filtering config
            filtering_config = LegacyFilteringConfig()
            
            # Create legacy extraction config
            extraction_config = LegacyExtractionConfig(
                extract_transcripts=unified_config.content_extraction.extract_text,
                extract_thumbnails=unified_config.content_extraction.download_images,
                extract_comments=False  # Default
            )
            
            # Create legacy performance config
            performance_config = LegacyPerformanceConfig(
                max_concurrent_requests=unified_config.performance.max_concurrent_requests,
                timeout=unified_config.performance.request_timeout,
                max_retries=unified_config.performance.max_retries,
                batch_size=unified_config.performance.batch_size
            )
            
            # Create legacy security config
            security_config = LegacySecurityConfig()
            
            # Create legacy monitoring config
            monitoring_config = LegacyMonitoringConfig(
                enabled=unified_config.enable_metrics
            )
            
            # Create legacy crawler configuration
            legacy_config = LegacyCrawlerConfig(
                crawl_mode=CrawlMode.HYBRID,
                geographical_focus=GeographicalFocus.GLOBAL,
                api=api_config,
                scraping=scraping_config,
                database=database_config,
                cache=cache_config,
                filtering=filtering_config,
                extraction=extraction_config,
                performance=performance_config,
                security=security_config,
                monitoring=monitoring_config
            )
            
            logger.info("Successfully converted unified configuration to legacy YouTube format")
            return legacy_config
            
        except Exception as e:
            logger.error(f"Failed to convert unified configuration to legacy YouTube format: {e}")
            return None


class YouTubeConfigurationManager:
    """
    Configuration manager for the YouTube crawler that handles both 
    legacy and unified configuration systems.
    """
    
    def __init__(self, use_unified: bool = True):
        self.use_unified = use_unified and UNIFIED_CONFIG_AVAILABLE
        self.adapter = YouTubeConfigurationAdapter()
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
                               context: str = "youtube_crawler") -> Any:
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
                    # Create default YouTube crawler configuration
                    self.unified_config = self._create_default_unified_config()
                    
                logger.info("Loaded unified YouTube crawler configuration")
                return self.unified_config
                
            except Exception as e:
                logger.error(f"Failed to load unified configuration: {e}")
                # Fall back to legacy configuration
                self.use_unified = False
        
        # Use legacy configuration system
        if LEGACY_CONFIG_AVAILABLE:
            try:
                from .config import get_default_config
                self.legacy_config = get_default_config()
                logger.info("Loaded legacy YouTube crawler configuration")
                return self.legacy_config
            except Exception as e:
                logger.error(f"Failed to load legacy configuration: {e}")
        
        # Return minimal default configuration
        return self._create_minimal_config()
    
    def _create_default_unified_config(self) -> UnifiedCrawlerConfiguration:
        """Create default unified configuration for YouTube crawler."""
        # Create API credentials with environment variable fallback
        api_credentials = APICredentialsConfiguration(
            youtube_api_key=None,  # Should be set via environment or config file
            youtube_quota_limit=10000,
            youtube_requests_per_minute=60
        )
        
        # Create moderate stealth configuration
        stealth = StealthConfiguration(
            level=StealthLevel.MODERATE,
            rotate_user_agents=True,
            min_request_delay=1.0,
            max_request_delay=3.0,
            randomize_headers=True
        )
        
        # Create performance configuration optimized for YouTube
        performance = CrawlerPerformanceConfiguration(
            max_concurrent_requests=5,  # Conservative for API limits
            requests_per_second=1.0,    # Respect YouTube rate limits
            request_timeout=30,
            max_retries=3,
            batch_size=50
        )
        
        # Create content extraction configuration for video content
        content_extraction = ContentExtractionConfiguration(
            extract_text=True,          # Extract video transcripts
            extract_metadata=True,      # Extract video metadata
            extract_title=True,         # Extract video titles
            extract_links=True,         # Extract channel/playlist links
            download_images=False,      # Don't download thumbnails by default
            min_content_length=0,       # No minimum for video content
            language_detection=True
        )
        
        # Create cache configuration for API responses
        cache = CrawlerCacheConfiguration(
            enabled=True,
            response_ttl=3600,          # 1 hour for API responses
            content_ttl=86400,          # 24 hours for video content
            max_cache_size_mb=1024      # Larger cache for video metadata
        )
        
        return UnifiedCrawlerConfiguration(
            crawler_type=CrawlerType.YOUTUBE,
            crawler_name="youtube_crawler",
            api_credentials=api_credentials,
            stealth=stealth,
            performance=performance,
            content_extraction=content_extraction,
            cache=cache,
            output_format="json",
            save_parsed_content=True,
            enable_metrics=True
        )
    
    def _create_minimal_config(self) -> Dict[str, Any]:
        """Create minimal configuration as fallback."""
        return {
            "crawler_type": "youtube",
            "api_key": None,
            "max_concurrent_requests": 5,
            "requests_per_second": 1.0,
            "request_timeout": 30,
            "enable_stealth": True,
            "cache_responses": True,
            "extract_transcripts": True
        }
    
    def get_configuration(self) -> Any:
        """Get the current configuration object."""
        return self.unified_config if self.use_unified else self.legacy_config
    
    def get_database_config(self) -> Optional[DatabaseConfiguration]:
        """Get database configuration from current config."""
        if self.use_unified and self.unified_config:
            # Get database config from global manager
            if self._config_manager:
                return self._config_manager.get_configuration_typed(
                    "youtube_crawler", "database", DatabaseConfiguration
                )
        elif self.legacy_config and hasattr(self.legacy_config, 'database'):
            # Convert legacy database config
            legacy_db = self.legacy_config.database
            return DatabaseConfiguration(
                host=legacy_db.host,
                port=legacy_db.port,
                database=legacy_db.database,
                username=legacy_db.username,
                password=legacy_db.password
            )
        
        return None
    
    def is_using_unified_config(self) -> bool:
        """Check if using unified configuration system."""
        return self.use_unified


# Convenience functions for backward compatibility
async def create_youtube_crawler_config(config_file: Optional[str] = None,
                                       use_unified: bool = True) -> Any:
    """
    Create YouTube crawler configuration.
    
    Args:
        config_file: Path to configuration file
        use_unified: Whether to use unified configuration system
        
    Returns:
        Configuration object
    """
    manager = YouTubeConfigurationManager(use_unified=use_unified)
    await manager.initialize()
    return await manager.load_configuration(config_file)


def get_youtube_crawler_config_manager(use_unified: bool = True) -> YouTubeConfigurationManager:
    """
    Get YouTube crawler configuration manager.
    
    Args:
        use_unified: Whether to use unified configuration system
        
    Returns:
        YouTubeConfigurationManager instance
    """
    return YouTubeConfigurationManager(use_unified=use_unified)


# Export all public classes and functions
__all__ = [
    'YouTubeConfigurationAdapter',
    'YouTubeConfigurationManager',
    'create_youtube_crawler_config',
    'get_youtube_crawler_config_manager',
    'UNIFIED_CONFIG_AVAILABLE',
    'LEGACY_CONFIG_AVAILABLE'
]