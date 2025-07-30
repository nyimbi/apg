"""
Generation Crawler Configuration Management
==========================================

Configuration system for the gen_crawler package providing
type-safe configuration management with validation.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class ContentFilterConfig:
    """Configuration for content filtering."""
    min_content_length: int = 100
    exclude_extensions: List[str] = field(default_factory=lambda: ['.pdf', '.doc', '.xls', '.zip'])
    include_patterns: List[str] = field(default_factory=lambda: ['article', 'news', 'post', 'story'])
    exclude_patterns: List[str] = field(default_factory=lambda: ['tag', 'category', 'archive', 'login'])
    max_content_length: int = 1000000  # 1MB limit
    allowed_content_types: List[str] = field(default_factory=lambda: ['text/html', 'application/xhtml+xml'])

@dataclass
class DatabaseConfig:
    """Database configuration for gen_crawler."""
    enable_database: bool = False
    connection_string: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    table_prefix: str = "gen_crawler_"

@dataclass
class PerformanceConfig:
    """Performance and concurrency configuration."""
    max_pages_per_site: int = 500
    max_concurrent: int = 5
    request_timeout: int = 30
    max_retries: int = 3
    crawl_delay: float = 2.0
    max_depth: int = 10
    rate_limit_delay: float = 1.0
    memory_limit_mb: int = 1024

@dataclass
class AdaptiveConfig:
    """Configuration for adaptive crawling behavior."""
    enable_adaptive_crawling: bool = True
    strategy_switching_threshold: float = 0.8  # Success rate threshold
    performance_monitoring: bool = True
    auto_optimize: bool = True
    strategy_evaluation_window: int = 100  # Pages to evaluate strategy
    min_pages_for_optimization: int = 50

@dataclass
class StealthConfig:
    """Stealth and anti-detection configuration."""
    enable_stealth: bool = True
    user_agent: str = 'GenCrawler/1.0 (+https://datacraft.co.ke)'
    random_user_agents: bool = True
    enable_proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)
    enable_request_headers_rotation: bool = True
    respect_robots_txt: bool = True

@dataclass
class GenCrawlerSettings:
    """Complete settings for GenCrawler."""
    content_filters: ContentFilterConfig = field(default_factory=ContentFilterConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    stealth: StealthConfig = field(default_factory=StealthConfig)
    
    # Additional settings
    enable_content_analysis: bool = True
    enable_image_extraction: bool = True
    enable_link_analysis: bool = True
    save_raw_html: bool = False
    compression_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'content_filters': {
                'min_content_length': self.content_filters.min_content_length,
                'exclude_extensions': self.content_filters.exclude_extensions,
                'include_patterns': self.content_filters.include_patterns,
                'exclude_patterns': self.content_filters.exclude_patterns,
                'max_content_length': self.content_filters.max_content_length,
                'allowed_content_types': self.content_filters.allowed_content_types
            },
            'database': {
                'enable_database': self.database.enable_database,
                'connection_string': self.database.connection_string,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow,
                'pool_timeout': self.database.pool_timeout,
                'pool_recycle': self.database.pool_recycle,
                'table_prefix': self.database.table_prefix
            },
            'performance': {
                'max_pages_per_site': self.performance.max_pages_per_site,
                'max_concurrent': self.performance.max_concurrent,
                'request_timeout': self.performance.request_timeout,
                'max_retries': self.performance.max_retries,
                'crawl_delay': self.performance.crawl_delay,
                'max_depth': self.performance.max_depth,
                'rate_limit_delay': self.performance.rate_limit_delay,
                'memory_limit_mb': self.performance.memory_limit_mb
            },
            'adaptive': {
                'enable_adaptive_crawling': self.adaptive.enable_adaptive_crawling,
                'strategy_switching_threshold': self.adaptive.strategy_switching_threshold,
                'performance_monitoring': self.adaptive.performance_monitoring,
                'auto_optimize': self.adaptive.auto_optimize,
                'strategy_evaluation_window': self.adaptive.strategy_evaluation_window,
                'min_pages_for_optimization': self.adaptive.min_pages_for_optimization
            },
            'stealth': {
                'enable_stealth': self.stealth.enable_stealth,
                'user_agent': self.stealth.user_agent,
                'random_user_agents': self.stealth.random_user_agents,
                'enable_proxy_rotation': self.stealth.enable_proxy_rotation,
                'proxy_list': self.stealth.proxy_list,
                'enable_request_headers_rotation': self.stealth.enable_request_headers_rotation,
                'respect_robots_txt': self.stealth.respect_robots_txt
            },
            'enable_content_analysis': self.enable_content_analysis,
            'enable_image_extraction': self.enable_image_extraction,
            'enable_link_analysis': self.enable_link_analysis,
            'save_raw_html': self.save_raw_html,
            'compression_enabled': self.compression_enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenCrawlerSettings':
        """Create settings from dictionary."""
        content_filters = ContentFilterConfig(**data.get('content_filters', {}))
        database = DatabaseConfig(**data.get('database', {}))
        performance = PerformanceConfig(**data.get('performance', {}))
        adaptive = AdaptiveConfig(**data.get('adaptive', {}))
        stealth = StealthConfig(**data.get('stealth', {}))
        
        return cls(
            content_filters=content_filters,
            database=database,
            performance=performance,
            adaptive=adaptive,
            stealth=stealth,
            enable_content_analysis=data.get('enable_content_analysis', True),
            enable_image_extraction=data.get('enable_image_extraction', True),
            enable_link_analysis=data.get('enable_link_analysis', True),
            save_raw_html=data.get('save_raw_html', False),
            compression_enabled=data.get('compression_enabled', True)
        )

class GenCrawlerConfig:
    """Configuration manager for GenCrawler."""
    
    def __init__(self, settings: Optional[GenCrawlerSettings] = None, 
                 config_file: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            settings: Pre-configured settings
            config_file: Path to configuration file
        """
        if settings:
            self.settings = settings
        elif config_file:
            self.settings = self._load_from_file(config_file)
        else:
            self.settings = self._get_default_settings()
        
        self.config_file = config_file
        self._validate_settings()
    
    def _get_default_settings(self) -> GenCrawlerSettings:
        """Get default settings."""
        return GenCrawlerSettings()
    
    def _load_from_file(self, config_file: Union[str, Path]) -> GenCrawlerSettings:
        """Load settings from configuration file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file {config_file} not found, using defaults")
            return self._get_default_settings()
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return GenCrawlerSettings.from_dict(data)
            
        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            return self._get_default_settings()
    
    def _validate_settings(self):
        """Validate configuration settings."""
        # Validate performance settings
        if self.settings.performance.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        
        if self.settings.performance.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        
        if self.settings.performance.max_pages_per_site <= 0:
            raise ValueError("max_pages_per_site must be positive")
        
        # Validate content filter settings
        if self.settings.content_filters.min_content_length < 0:
            raise ValueError("min_content_length cannot be negative")
        
        if self.settings.content_filters.max_content_length <= self.settings.content_filters.min_content_length:
            raise ValueError("max_content_length must be greater than min_content_length")
        
        # Validate database settings
        if self.settings.database.enable_database and not self.settings.database.connection_string:
            logger.warning("Database enabled but no connection string provided")
        
        logger.debug("Configuration validation passed")
    
    def save_to_file(self, config_file: Union[str, Path]):
        """Save current settings to file."""
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.settings.to_dict(), f, indent=2)
            
            logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config to {config_file}: {e}")
            raise
    
    def update_from_env(self):
        """Update settings from environment variables."""
        env_mappings = {
            'GEN_CRAWLER_MAX_PAGES': ('performance', 'max_pages_per_site', int),
            'GEN_CRAWLER_MAX_CONCURRENT': ('performance', 'max_concurrent', int),
            'GEN_CRAWLER_REQUEST_TIMEOUT': ('performance', 'request_timeout', int),
            'GEN_CRAWLER_CRAWL_DELAY': ('performance', 'crawl_delay', float),
            'GEN_CRAWLER_USER_AGENT': ('stealth', 'user_agent', str),
            'GEN_CRAWLER_ENABLE_DATABASE': ('database', 'enable_database', lambda x: x.lower() == 'true'),
            'GEN_CRAWLER_DB_CONNECTION': ('database', 'connection_string', str),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    section_obj = getattr(self.settings, section)
                    setattr(section_obj, key, converted_value)
                    logger.debug(f"Updated {section}.{key} from environment: {converted_value}")
                except Exception as e:
                    logger.warning(f"Error converting environment variable {env_var}={value}: {e}")
    
    def get_crawler_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for GenCrawler."""
        return {
            'max_pages_per_site': self.settings.performance.max_pages_per_site,
            'max_concurrent': self.settings.performance.max_concurrent,
            'request_timeout': self.settings.performance.request_timeout,
            'max_retries': self.settings.performance.max_retries,
            'enable_database': self.settings.database.enable_database,
            'database_config': {
                'connection_string': self.settings.database.connection_string,
                'pool_size': self.settings.database.pool_size,
                'max_overflow': self.settings.database.max_overflow
            } if self.settings.database.enable_database else {},
            'enable_content_analysis': self.settings.enable_content_analysis,
            'respect_robots_txt': self.settings.stealth.respect_robots_txt,
            'crawl_delay': self.settings.performance.crawl_delay,
            'user_agent': self.settings.stealth.user_agent,
            'enable_adaptive_crawling': self.settings.adaptive.enable_adaptive_crawling,
            'max_depth': self.settings.performance.max_depth,
            'content_filters': {
                'min_content_length': self.settings.content_filters.min_content_length,
                'exclude_extensions': self.settings.content_filters.exclude_extensions,
                'include_patterns': self.settings.content_filters.include_patterns,
                'exclude_patterns': self.settings.content_filters.exclude_patterns
            }
        }
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """Get configuration for AdaptiveCrawler."""
        return {
            'enable_adaptive_crawling': self.settings.adaptive.enable_adaptive_crawling,
            'strategy_switching_threshold': self.settings.adaptive.strategy_switching_threshold,
            'performance_monitoring': self.settings.adaptive.performance_monitoring,
            'auto_optimize': self.settings.adaptive.auto_optimize,
            'strategy_evaluation_window': self.settings.adaptive.strategy_evaluation_window,
            'min_pages_for_optimization': self.settings.adaptive.min_pages_for_optimization
        }

def create_gen_config(config_file: Optional[Union[str, Path]] = None,
                     settings: Optional[GenCrawlerSettings] = None,
                     load_from_env: bool = True) -> GenCrawlerConfig:
    """
    Factory function to create GenCrawlerConfig.
    
    Args:
        config_file: Path to configuration file
        settings: Pre-configured settings
        load_from_env: Whether to load from environment variables
        
    Returns:
        Configured GenCrawlerConfig instance
    """
    config = GenCrawlerConfig(settings=settings, config_file=config_file)
    
    if load_from_env:
        config.update_from_env()
    
    return config

def get_default_gen_config() -> Dict[str, Any]:
    """Get default configuration dictionary."""
    settings = GenCrawlerSettings()
    config = GenCrawlerConfig(settings=settings)
    return config.get_crawler_config()

def load_gen_config_from_file(config_file: Union[str, Path]) -> GenCrawlerConfig:
    """Load configuration from file."""
    return GenCrawlerConfig(config_file=config_file)

def create_gen_config_from_dict(config_dict: Dict[str, Any]) -> GenCrawlerConfig:
    """Create configuration from dictionary."""
    settings = GenCrawlerSettings.from_dict(config_dict)
    return GenCrawlerConfig(settings=settings)