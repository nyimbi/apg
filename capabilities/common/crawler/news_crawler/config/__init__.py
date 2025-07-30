"""
News Crawler Configuration Module
==================================

Unified configuration management for the news crawler system using utils/config infrastructure.

Components:
- NewsConfigurationManager: Unified configuration manager
- NewsConfigurationAdapter: Adapter for configuration conversion

Factory Functions:
- create_news_crawler_config(): Create unified news crawler configuration
- get_news_crawler_config_manager(): Get configuration manager

Environment Support:
- Configuration can be loaded from environment variables
- Supports validation and defaults
- Uses utils/config infrastructure

Author: Lindela Development Team
Version: 4.0.0
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
import logging

# Version information
__version__ = "4.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Configure logging
logger = logging.getLogger(__name__)

# Import unified configuration system
try:
    from .unified_config import (
        NewsConfigurationManager,
        NewsConfigurationAdapter,
        create_news_crawler_config,
        get_news_crawler_config_manager,
        UNIFIED_CONFIG_AVAILABLE,
        LEGACY_CONFIG_AVAILABLE
    )
    _UNIFIED_CONFIG_AVAILABLE = True
    logger.debug("Unified configuration system loaded")
except ImportError as e:
    logger.warning(f"Unified configuration system not available: {e}")
    _UNIFIED_CONFIG_AVAILABLE = False
    
    # Placeholder classes
    class NewsConfigurationManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("Unified configuration not available")
    
    class NewsConfigurationAdapter:
        def __init__(self, *args, **kwargs):
            raise ImportError("Unified configuration not available")
    
    def create_news_crawler_config(*args, **kwargs):
        raise ImportError("Unified configuration not available")
    
    def get_news_crawler_config_manager(*args, **kwargs):
        raise ImportError("Unified configuration not available")
    
    UNIFIED_CONFIG_AVAILABLE = False
    LEGACY_CONFIG_AVAILABLE = False


# Default configurations for fallback
DEFAULT_CRAWLER_CONFIG = {
    'max_concurrent_requests': 5,
    'requests_per_second': 2.0,
    'request_timeout': 30,
    'max_retries': 3,
    'enable_stealth': True,
    'enable_bypass': True,
    'enable_enhanced_stealth': True,
    'enable_ml_analysis': True,
    'cache_responses': True,
    'save_parsed_content': True,
    'output_format': 'json'
}


class ConfigurationManager:
    """Simple configuration manager for news crawler when unified config not available."""
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file
        self._config = DEFAULT_CRAWLER_CONFIG.copy()
        self._load_env_config()
    
    def _load_env_config(self):
        """Load configuration from environment variables."""
        import os
        
        env_mappings = {
            'NEWS_CRAWLER_MAX_CONCURRENT': ('max_concurrent_requests', int),
            'NEWS_CRAWLER_TIMEOUT': ('request_timeout', int),
            'NEWS_CRAWLER_RETRIES': ('max_retries', int),
            'NEWS_CRAWLER_ENABLE_STEALTH': ('enable_stealth', bool),
            'NEWS_CRAWLER_ENABLE_ML': ('enable_ml_analysis', bool),
            'NEWS_CRAWLER_RATE_LIMIT': ('requests_per_second', float)
        }
        
        for env_var, (config_key, config_type) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    if config_type == bool:
                        self._config[config_key] = value.lower() in ('true', '1', 'yes')
                    elif config_type == int:
                        self._config[config_key] = int(value)
                    elif config_type == float:
                        self._config[config_key] = float(value)
                    else:
                        self._config[config_key] = value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> Union[NewsConfigurationManager, ConfigurationManager]:
    """Get configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        if _UNIFIED_CONFIG_AVAILABLE:
            _config_manager = NewsConfigurationManager()
        else:
            _config_manager = ConfigurationManager()
    return _config_manager


def get_crawler_config(**kwargs) -> Dict[str, Any]:
    """Get crawler configuration with optional overrides."""
    manager = get_config_manager()
    if hasattr(manager, 'get_configuration'):
        # Unified config manager
        config = manager.get_configuration()
        if hasattr(config, 'to_dict'):
            base_config = config.to_dict()
        else:
            base_config = DEFAULT_CRAWLER_CONFIG.copy()
    else:
        # Simple config manager
        base_config = manager.to_dict()
    
    base_config.update(kwargs)
    return base_config


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary."""
    required_fields = ['max_concurrent_requests', 'request_timeout', 'max_retries']
    
    for field in required_fields:
        if field not in config:
            logger.warning(f"Missing configuration field: {field}, using default")
    
    # Validate ranges with warnings rather than errors
    max_concurrent = config.get('max_concurrent_requests', 5)
    if max_concurrent <= 0 or max_concurrent > 100:
        logger.warning(f"max_concurrent_requests ({max_concurrent}) should be between 1 and 100")
    
    timeout = config.get('request_timeout', 30)
    if timeout <= 0 or timeout > 300:
        logger.warning(f"request_timeout ({timeout}) should be between 1 and 300 seconds")
    
    retries = config.get('max_retries', 3)
    if retries < 0 or retries > 10:
        logger.warning(f"max_retries ({retries}) should be between 0 and 10")
    
    return True


def get_config_health() -> Dict[str, Any]:
    """Get health status of configuration system."""
    return {
        'status': 'healthy' if _UNIFIED_CONFIG_AVAILABLE else 'degraded',
        'unified_config_available': _UNIFIED_CONFIG_AVAILABLE,
        'version': __version__,
        'config_loaded': _config_manager is not None
    }


# Export all public components
__all__ = [
    # Configuration classes
    'NewsConfigurationManager',
    'NewsConfigurationAdapter',
    'ConfigurationManager',
    
    # Factory functions
    'create_news_crawler_config',
    'get_news_crawler_config_manager',
    
    # Default configurations
    'DEFAULT_CRAWLER_CONFIG',
    
    # Utility functions
    'get_config_manager',
    'get_crawler_config',
    'validate_config',
    'get_config_health',
    
    # Flags
    'UNIFIED_CONFIG_AVAILABLE',
    'LEGACY_CONFIG_AVAILABLE',
    
    # Version info
    '__version__',
    '__author__',
    '__license__'
]

# Module initialization
logger.info(f"News Crawler Configuration Module v{__version__} initialized")
logger.info(f"Unified configuration available: {_UNIFIED_CONFIG_AVAILABLE}")