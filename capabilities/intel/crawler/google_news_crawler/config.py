"""
Configuration Management for Google News Crawler
===============================================

Comprehensive configuration system for the enhanced Google News crawler,
supporting multiple configuration sources, validation, and environment-specific settings.

Features:
- Multiple configuration sources (files, environment variables, defaults)
- Configuration validation and type checking
- Environment-specific configurations
- Dynamic configuration updates
- Configuration templates
- Secure credential management

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class CacheType(Enum):
    """Cache types."""
    MEMORY = "memory"
    REDIS = "redis"
    FILESYSTEM = "filesystem"
    DISABLED = "disabled"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "lindela"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False

    def get_connection_url(self) -> str:
        """Get database connection URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class CacheConfig:
    """Cache configuration."""
    type: CacheType = CacheType.MEMORY
    redis_url: str = "redis://localhost:6379/0"
    filesystem_path: str = "/tmp/lindela_cache"
    default_ttl: int = 3600  # 1 hour
    max_size: int = 1000
    enable_compression: bool = True

@dataclass
class StealthConfig:
    """Stealth crawling configuration."""
    enabled: bool = True
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    ])
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: int = 30
    concurrent_requests: int = 5
    enable_cloudflare_bypass: bool = True
    rotate_user_agents: bool = True
    respect_robots_txt: bool = True

@dataclass
class FilteringConfig:
    """Content filtering configuration."""
    min_content_length: int = 100
    max_content_length: int = 50000
    min_title_length: int = 10
    max_title_length: int = 500
    min_authority_score: float = 0.3
    min_reliability_score: float = 0.4
    min_readability_score: float = 0.2
    allowed_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de", "it"])
    blocked_domains: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    content_quality_threshold: float = 0.5

@dataclass
class ParsingConfig:
    """Parsing configuration."""
    max_articles_per_feed: int = 100
    extract_images: bool = True
    extract_links: bool = False
    clean_html: bool = True
    calculate_readability: bool = True
    enable_sentiment_analysis: bool = False
    enable_language_detection: bool = True
    enable_ml_features: bool = False
    min_confidence_score: float = 0.6

@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    max_concurrent_requests: int = 20
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_caching: bool = True
    enable_compression: bool = True
    enable_monitoring: bool = True
    batch_size: int = 50

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_console: bool = True
    enable_file: bool = False

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = True
    prometheus_port: int = 8000
    metrics_path: str = "/metrics"
    enable_health_checks: bool = True
    health_check_interval: int = 60
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "error_rate": 0.1,
        "response_time": 5.0,
        "success_rate": 0.95
    })

@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key: Optional[str] = None
    rate_limit_requests: int = 1000
    rate_limit_window: int = 3600  # 1 hour
    enable_ip_whitelist: bool = False
    allowed_ips: List[str] = field(default_factory=list)
    encrypt_sensitive_data: bool = True
    hash_algorithm: str = "sha256"

@dataclass
class GoogleNewsConfig:
    """Google News specific configuration."""
    base_url: str = "https://news.google.com/rss"
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh", "ar", "hi"
    ])
    supported_countries: List[str] = field(default_factory=lambda: [
        "US", "GB", "DE", "FR", "IT", "ES", "CA", "AU", "IN", "BR", "JP", "KR", "CN"
    ])
    max_results_per_query: int = 100
    default_language: str = "en"
    default_country: str = "US"
    enable_topic_feeds: bool = True
    enable_location_feeds: bool = True
    enable_site_specific_feeds: bool = True

@dataclass
class NewsSourceConfig:
    """News source configuration."""
    rss_feeds: List[str] = field(default_factory=list)
    discovery_enabled: bool = True
    validation_enabled: bool = True
    health_check_interval: int = 3600  # 1 hour
    max_failed_attempts: int = 5
    retry_backoff_factor: float = 2.0

@dataclass
class CrawlerConfig:
    """Main crawler configuration."""
    # Core configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    stealth: StealthConfig = field(default_factory=StealthConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    google_news: GoogleNewsConfig = field(default_factory=GoogleNewsConfig)
    news_sources: NewsSourceConfig = field(default_factory=NewsSourceConfig)

    # Environment
    environment: str = "development"
    debug: bool = False
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CrawlerConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        config_data = data.copy()

        # Convert nested dictionaries to dataclass instances
        nested_configs = {
            'database': DatabaseConfig,
            'cache': CacheConfig,
            'stealth': StealthConfig,
            'filtering': FilteringConfig,
            'parsing': ParsingConfig,
            'performance': PerformanceConfig,
            'logging': LoggingConfig,
            'monitoring': MonitoringConfig,
            'security': SecurityConfig,
            'google_news': GoogleNewsConfig,
            'news_sources': NewsSourceConfig,
        }

        for key, config_class in nested_configs.items():
            if key in config_data and isinstance(config_data[key], dict):
                config_data[key] = config_class(**config_data[key])

        return cls(**config_data)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []

        # Database validation
        if not self.database.host:
            errors.append("Database host is required")
        if not self.database.database:
            errors.append("Database name is required")
        if self.database.port <= 0 or self.database.port > 65535:
            errors.append("Database port must be between 1 and 65535")

        # Performance validation
        if self.performance.max_concurrent_requests <= 0:
            errors.append("Max concurrent requests must be positive")
        if self.performance.request_timeout <= 0:
            errors.append("Request timeout must be positive")

        # Filtering validation
        if self.filtering.min_content_length < 0:
            errors.append("Min content length cannot be negative")
        if self.filtering.max_content_length < self.filtering.min_content_length:
            errors.append("Max content length must be greater than min content length")

        # Google News validation
        if not self.google_news.base_url:
            errors.append("Google News base URL is required")
        if self.google_news.default_language not in self.google_news.supported_languages:
            errors.append("Default language must be in supported languages")
        if self.google_news.default_country not in self.google_news.supported_countries:
            errors.append("Default country must be in supported countries")

        return errors

class ConfigurationManager:
    """Configuration manager for the Google News crawler."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self._config: Optional[CrawlerConfig] = None
        self._config_watchers: List[callable] = []

    def load_config(self, config_path: Optional[str] = None) -> CrawlerConfig:
        """Load configuration from various sources."""
        config_path = config_path or self.config_path

        # Start with default configuration
        config_data = {}

        # Load from file if provided
        if config_path and os.path.exists(config_path):
            config_data = self._load_config_file(config_path)

        # Override with environment variables
        env_config = self._load_from_environment()
        config_data = self._merge_configs(config_data, env_config)

        # Create configuration object
        config = CrawlerConfig.from_dict(config_data)

        # Validate configuration
        errors = config.validate()
        if errors:
            raise ConfigurationError(f"Configuration validation failed: {'; '.join(errors)}")

        self._config = config
        logger.info(f"Configuration loaded successfully from {config_path or 'defaults'}")

        return config

    def get_config(self) -> CrawlerConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config

    def save_config(self, config: CrawlerConfig, config_path: str, format: str = "yaml") -> None:
        """Save configuration to file."""
        config_data = config.to_dict()

        if format.lower() == "yaml":
            if not YAML_AVAILABLE:
                raise ConfigurationError("PyYAML not available for YAML format")
            with open(config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
        else:
            raise ConfigurationError(f"Unsupported format: {format}")

        logger.info(f"Configuration saved to {config_path}")

    def create_template(self, template_path: str, format: str = "yaml") -> None:
        """Create configuration template file."""
        template_config = CrawlerConfig()
        self.save_config(template_config, template_path, format)
        logger.info(f"Configuration template created at {template_path}")

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    if not YAML_AVAILABLE:
                        raise ConfigurationError("PyYAML not available for YAML files")
                    return yaml.safe_load(f) or {}
                elif config_path.endswith('.json'):
                    return json.load(f) or {}
                else:
                    raise ConfigurationError(f"Unsupported config file format: {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {config_path}: {e}")

    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}

        # Environment variable mappings
        env_mappings = {
            'LINDELA_DB_HOST': 'database.host',
            'LINDELA_DB_PORT': 'database.port',
            'LINDELA_DB_NAME': 'database.database',
            'LINDELA_DB_USER': 'database.username',
            'LINDELA_DB_PASSWORD': 'database.password',
            'LINDELA_CACHE_TYPE': 'cache.type',
            'LINDELA_REDIS_URL': 'cache.redis_url',
            'LINDELA_LOG_LEVEL': 'logging.level',
            'LINDELA_LOG_FILE': 'logging.file_path',
            'LINDELA_ENVIRONMENT': 'environment',
            'LINDELA_DEBUG': 'debug',
            'LINDELA_API_KEY': 'security.api_key',
            'LINDELA_ENABLE_STEALTH': 'stealth.enabled',
            'LINDELA_MAX_CONCURRENT': 'performance.max_concurrent_requests',
        }

        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(env_config, config_path, self._convert_env_value(value))

        return env_config

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable value to appropriate type."""
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # Integer conversion
        if value.isdigit():
            return int(value)

        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # String value
        return value

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested configuration value using dot notation."""
        keys = path.split('.')
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def add_config_watcher(self, callback: callable) -> None:
        """Add configuration change watcher."""
        self._config_watchers.append(callback)

    def remove_config_watcher(self, callback: callable) -> None:
        """Remove configuration change watcher."""
        if callback in self._config_watchers:
            self._config_watchers.remove(callback)

    def notify_config_change(self) -> None:
        """Notify all watchers of configuration change."""
        for callback in self._config_watchers:
            try:
                callback(self._config)
            except Exception as e:
                logger.error(f"Config watcher callback failed: {e}")

# Global configuration manager instance
config_manager = ConfigurationManager()

def get_config() -> CrawlerConfig:
    """Get global configuration."""
    return config_manager.get_config()

def load_config(config_path: str) -> CrawlerConfig:
    """Load configuration from file."""
    return config_manager.load_config(config_path)

def create_config_template(template_path: str, format: str = "yaml") -> None:
    """Create configuration template."""
    config_manager.create_template(template_path, format)

# Configuration presets for different environments
DEVELOPMENT_CONFIG = {
    'environment': 'development',
    'debug': True,
    'logging': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': True,
    },
    'performance': {
        'max_concurrent_requests': 5,
        'enable_caching': True,
    },
    'monitoring': {
        'enabled': True,
        'enable_health_checks': True,
    }
}

PRODUCTION_CONFIG = {
    'environment': 'production',
    'debug': False,
    'logging': {
        'level': 'INFO',
        'enable_console': False,
        'enable_file': True,
    },
    'performance': {
        'max_concurrent_requests': 20,
        'enable_caching': True,
        'enable_compression': True,
    },
    'monitoring': {
        'enabled': True,
        'enable_health_checks': True,
        'prometheus_port': 8000,
    },
    'security': {
        'rate_limit_requests': 1000,
        'encrypt_sensitive_data': True,
    }
}

TESTING_CONFIG = {
    'environment': 'testing',
    'debug': True,
    'database': {
        'database': 'lindela_test',
    },
    'cache': {
        'type': 'memory',
    },
    'logging': {
        'level': 'DEBUG',
        'enable_console': True,
        'enable_file': False,
    },
    'performance': {
        'max_concurrent_requests': 2,
        'enable_caching': False,
    },
    'monitoring': {
        'enabled': False,
    }
}

def get_preset_config(preset: str) -> Dict[str, Any]:
    """Get preset configuration."""
    presets = {
        'development': DEVELOPMENT_CONFIG,
        'production': PRODUCTION_CONFIG,
        'testing': TESTING_CONFIG,
    }

    if preset not in presets:
        raise ConfigurationError(f"Unknown preset: {preset}")

    return presets[preset]
