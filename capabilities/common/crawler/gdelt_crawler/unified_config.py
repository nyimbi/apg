"""
Unified Configuration Adapter for GDELT Crawler
===============================================

Adapter that bridges the GDELT crawler's configuration system
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

logger = logging.getLogger(__name__)


class GDELTConfigurationManager:
    """Configuration manager for the GDELT crawler."""
    
    def __init__(self, use_unified: bool = True):
        self.use_unified = use_unified and UNIFIED_CONFIG_AVAILABLE
        self.unified_config: Optional[UnifiedCrawlerConfiguration] = None
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
        """Create default unified configuration for GDELT crawler."""
        # Create API credentials (GDELT doesn't require API keys)
        api_credentials = APICredentialsConfiguration()
        
        # Create basic stealth configuration
        stealth = StealthConfiguration(
            level=StealthLevel.BASIC,
            rotate_user_agents=True,
            min_request_delay=1.0,
            max_request_delay=2.0
        )
        
        # Create performance configuration for bulk processing
        performance = CrawlerPerformanceConfiguration(
            max_concurrent_requests=10,
            requests_per_second=5.0,
            request_timeout=60,
            max_retries=3,
            batch_size=1000  # Large batches for GDELT data
        )
        
        # Create content extraction configuration
        content_extraction = ContentExtractionConfiguration(
            extract_text=True,
            extract_metadata=True,
            min_content_length=0  # Accept all GDELT records
        )
        
        # Create cache configuration for GDELT files
        cache = CrawlerCacheConfiguration(
            enabled=True,
            response_ttl=3600,      # 1 hour for file listings
            content_ttl=86400       # 24 hours for file content
        )
        
        return UnifiedCrawlerConfiguration(
            crawler_type=CrawlerType.GDELT,
            crawler_name="gdelt_crawler",
            api_credentials=api_credentials,
            stealth=stealth,
            performance=performance,
            content_extraction=content_extraction,
            cache=cache
        )


# Export classes
__all__ = [
    'GDELTConfigurationManager',
    'UNIFIED_CONFIG_AVAILABLE'
]