"""
YouTube Crawler Configuration Module
====================================

Comprehensive configuration management for YouTube crawler operations.
Supports multiple environments, API configurations, and crawling parameters.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class CrawlMode(Enum):
    """Crawling mode enumeration."""
    API_ONLY = "api_only"
    SCRAPING_ONLY = "scraping_only"
    HYBRID = "hybrid"
    AUTO = "auto"

class VideoQuality(Enum):
    """Video quality enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"

class ContentType(Enum):
    """Content type enumeration."""
    VIDEO = "video"
    CHANNEL = "channel"
    PLAYLIST = "playlist"
    COMMENT = "comment"
    LIVE_STREAM = "live_stream"
    SHORT = "short"

class GeographicalFocus(Enum):
    """Geographical focus enumeration."""
    GLOBAL = "global"
    US = "us"
    GB = "gb"
    CA = "ca"
    AU = "au"
    IN = "in"
    DE = "de"
    FR = "fr"
    JP = "jp"
    BR = "br"
    CUSTOM = "custom"

@dataclass
class APIConfig:
    """YouTube API configuration."""
    api_key: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    quota_limit: int = 10000
    requests_per_minute: int = 60
    requests_per_day: int = 1000000
    enable_quota_monitoring: bool = True
    fallback_to_scraping: bool = True

@dataclass
class ScrapingConfig:
    """Web scraping configuration."""
    enable_stealth: bool = True
    user_agents: List[str] = field(default_factory=lambda: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ])
    request_delay: float = 1.0
    random_delay: bool = True
    max_retries: int = 3
    timeout: int = 30
    proxy_rotation: bool = False
    proxy_list: List[str] = field(default_factory=list)
    respect_robots_txt: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "lindela"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    max_connections: int = 20
    connection_timeout: int = 30
    query_timeout: int = 60
    auto_commit: bool = True
    enable_logging: bool = False

@dataclass
class CacheConfig:
    """Caching configuration."""
    enable_caching: bool = True
    cache_backend: str = "memory"  # memory, redis, file
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1000  # items
    cache_path: Optional[str] = None
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

@dataclass
class FilteringConfig:
    """Content filtering configuration."""
    min_video_duration: int = 30  # seconds
    max_video_duration: int = 7200  # seconds (2 hours)
    min_view_count: int = 0
    max_view_count: Optional[int] = None
    min_subscriber_count: int = 0
    allowed_languages: List[str] = field(default_factory=lambda: ["en"])
    blocked_keywords: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    content_categories: List[str] = field(default_factory=list)
    age_restriction: bool = False
    quality_threshold: float = 0.0  # 0-1 score

@dataclass
class ExtractionConfig:
    """Data extraction configuration."""
    extract_transcripts: bool = True
    extract_comments: bool = True
    extract_thumbnails: bool = True
    extract_metadata: bool = True
    extract_captions: bool = True
    max_comments: int = 100
    comment_sort_order: str = "relevance"  # relevance, time
    transcript_languages: List[str] = field(default_factory=lambda: ["en"])
    thumbnail_sizes: List[str] = field(default_factory=lambda: ["medium", "high"])
    include_video_stats: bool = True
    include_channel_info: bool = True

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    concurrent_requests: int = 5
    batch_size: int = 50
    enable_compression: bool = True
    memory_limit_mb: int = 512
    disk_cache_limit_mb: int = 1024
    enable_monitoring: bool = True
    log_performance_metrics: bool = True
    optimize_for_speed: bool = False
    optimize_for_accuracy: bool = True

@dataclass
class CrawlerConfig:
    """Main crawler configuration."""
    # Core settings
    crawl_mode: CrawlMode = CrawlMode.HYBRID
    geographical_focus: GeographicalFocus = GeographicalFocus.GLOBAL
    content_types: List[ContentType] = field(default_factory=lambda: [ContentType.VIDEO])

    # Component configurations
    api: APIConfig = field(default_factory=APIConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    enable_file_logging: bool = True
    enable_console_logging: bool = True

    # Output configuration
    output_format: str = "json"  # json, csv, parquet
    output_directory: str = "./youtube_data"
    save_raw_data: bool = False
    compress_output: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlerConfig':
        """Create configuration from dictionary."""
        # Handle enum conversions
        if 'crawl_mode' in data and isinstance(data['crawl_mode'], str):
            data['crawl_mode'] = CrawlMode(data['crawl_mode'])

        if 'geographical_focus' in data and isinstance(data['geographical_focus'], str):
            data['geographical_focus'] = GeographicalFocus(data['geographical_focus'])

        if 'content_types' in data and isinstance(data['content_types'], list):
            data['content_types'] = [ContentType(ct) if isinstance(ct, str) else ct for ct in data['content_types']]

        # Create nested configurations
        for key in ['api', 'scraping', 'database', 'cache', 'filtering', 'extraction', 'performance']:
            if key in data and isinstance(data[key], dict):
                config_class = {
                    'api': APIConfig,
                    'scraping': ScrapingConfig,
                    'database': DatabaseConfig,
                    'cache': CacheConfig,
                    'filtering': FilteringConfig,
                    'extraction': ExtractionConfig,
                    'performance': PerformanceConfig
                }[key]
                data[key] = config_class(**data[key])

        return cls(**data)

    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif filepath.endswith(('.yml', '.yaml')):
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'CrawlerConfig':
        """Load configuration from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.endswith(('.yml', '.yaml')):
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        return cls.from_dict(data)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Validate API configuration
        if self.crawl_mode in [CrawlMode.API_ONLY, CrawlMode.HYBRID]:
            if not self.api.api_key:
                errors.append("API key is required for API-based crawling")

        # Validate database configuration
        if not self.database.host:
            errors.append("Database host is required")
        if not self.database.database:
            errors.append("Database name is required")

        # Validate filtering configuration
        if self.filtering.min_video_duration > self.filtering.max_video_duration:
            errors.append("Minimum video duration cannot be greater than maximum")

        if self.filtering.max_view_count and self.filtering.min_view_count > self.filtering.max_view_count:
            errors.append("Minimum view count cannot be greater than maximum")

        # Validate performance configuration
        if self.performance.concurrent_requests <= 0:
            errors.append("Concurrent requests must be positive")
        if self.performance.batch_size <= 0:
            errors.append("Batch size must be positive")

        return errors

class ConfigurationManager:
    """Configuration management utility."""

    def __init__(self, config_dir: str = "./config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config_cache = {}

    def create_default_config(self) -> CrawlerConfig:
        """Create a default configuration."""
        return CrawlerConfig()

    def create_production_config(self) -> CrawlerConfig:
        """Create a production-optimized configuration."""
        config = CrawlerConfig()

        # Production optimizations
        config.performance.concurrent_requests = 10
        config.performance.batch_size = 100
        config.performance.optimize_for_speed = True
        config.cache.enable_caching = True
        config.cache.cache_ttl = 7200  # 2 hours
        config.log_level = "WARNING"
        config.database.connection_pool_size = 20

        return config

    def create_development_config(self) -> CrawlerConfig:
        """Create a development configuration."""
        config = CrawlerConfig()

        # Development settings
        config.performance.concurrent_requests = 2
        config.performance.batch_size = 10
        config.log_level = "DEBUG"
        config.save_raw_data = True
        config.extraction.max_comments = 20

        return config

    def create_testing_config(self) -> CrawlerConfig:
        """Create a testing configuration."""
        config = CrawlerConfig()

        # Testing settings
        config.performance.concurrent_requests = 1
        config.performance.batch_size = 5
        config.cache.enable_caching = False
        config.database.database = "lindela_test"
        config.filtering.max_video_duration = 300  # 5 minutes max for testing
        config.extraction.max_comments = 10

        return config

    def get_config(self, environment: str = "default") -> CrawlerConfig:
        """Get configuration for specified environment."""
        if environment in self._config_cache:
            return self._config_cache[environment]

        config_file = self.config_dir / f"{environment}.yml"

        if config_file.exists():
            config = CrawlerConfig.load_from_file(str(config_file))
        else:
            # Create default configurations
            if environment == "production":
                config = self.create_production_config()
            elif environment == "development":
                config = self.create_development_config()
            elif environment == "testing":
                config = self.create_testing_config()
            else:
                config = self.create_default_config()

            # Save the configuration
            config.save_to_file(str(config_file))

        # Validate configuration
        errors = config.validate()
        if errors:
            logger.warning(f"Configuration validation errors: {errors}")

        self._config_cache[environment] = config
        return config

    def update_config(self, environment: str, updates: Dict[str, Any]) -> CrawlerConfig:
        """Update configuration with new values."""
        config = self.get_config(environment)
        config_dict = config.to_dict()

        # Deep update
        self._deep_update(config_dict, updates)

        # Create new config and validate
        updated_config = CrawlerConfig.from_dict(config_dict)
        errors = updated_config.validate()
        if errors:
            raise ValueError(f"Invalid configuration updates: {errors}")

        # Save and cache
        config_file = self.config_dir / f"{environment}.yml"
        updated_config.save_to_file(str(config_file))
        self._config_cache[environment] = updated_config

        return updated_config

    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def list_configurations(self) -> List[str]:
        """List available configuration files."""
        return [f.stem for f in self.config_dir.glob("*.yml")]

    def delete_config(self, environment: str) -> bool:
        """Delete a configuration file."""
        config_file = self.config_dir / f"{environment}.yml"
        if config_file.exists():
            config_file.unlink()
            if environment in self._config_cache:
                del self._config_cache[environment]
            return True
        return False

# Global configuration manager instance
_config_manager = None

def get_config_manager(config_dir: str = "./config") -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigurationManager(config_dir)
    return _config_manager

def get_config(environment: str = None) -> CrawlerConfig:
    """Get configuration for the specified environment."""
    if environment is None:
        environment = os.getenv('YOUTUBE_CRAWLER_ENV', 'default')

    manager = get_config_manager()
    return manager.get_config(environment)

def load_config(filepath: str) -> CrawlerConfig:
    """Load configuration from file."""
    return CrawlerConfig.load_from_file(filepath)

def create_sample_config(filepath: str = "./config/example.yml") -> None:
    """Create a sample configuration file."""
    config = CrawlerConfig()
    config.save_to_file(filepath)
    print(f"Sample configuration created at: {filepath}")

# Environment variable overrides
def apply_env_overrides(config: CrawlerConfig) -> CrawlerConfig:
    """Apply environment variable overrides to configuration."""
    env_mappings = {
        'YOUTUBE_API_KEY': ('api', 'api_key'),
        'YOUTUBE_DB_HOST': ('database', 'host'),
        'YOUTUBE_DB_PORT': ('database', 'port'),
        'YOUTUBE_DB_NAME': ('database', 'database'),
        'YOUTUBE_DB_USER': ('database', 'username'),
        'YOUTUBE_DB_PASS': ('database', 'password'),
        'YOUTUBE_LOG_LEVEL': ('log_level',),
        'YOUTUBE_CACHE_TTL': ('cache', 'cache_ttl'),
        'YOUTUBE_CONCURRENT_REQUESTS': ('performance', 'concurrent_requests'),
        'YOUTUBE_BATCH_SIZE': ('performance', 'batch_size'),
    }

    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Navigate to the nested attribute
            obj = config
            for attr in config_path[:-1]:
                obj = getattr(obj, attr)

            # Convert value to appropriate type
            current_value = getattr(obj, config_path[-1])
            if isinstance(current_value, int):
                value = int(value)
            elif isinstance(current_value, float):
                value = float(value)
            elif isinstance(current_value, bool):
                value = value.lower() in ('true', '1', 'yes', 'on')

            setattr(obj, config_path[-1], value)

    return config
