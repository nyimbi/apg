#!/usr/bin/env python3
"""
NewsAPI Configuration Module
===========================

Configuration management for the NewsAPI crawler package.

This module provides:
- Configuration class for the NewsAPI client
- Functions for loading and saving configurations
- Default configuration settings

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class NewsAPIConfig:
    """Configuration for the NewsAPI client."""

    # API settings
    api_key: Optional[str] = None
    base_url: str = "https://newsapi.org/v2"
    
    # NewsData.io settings
    newsdata_api_key: Optional[str] = None
    newsdata_base_url: str = "https://newsdata.io/api/1"

    # Client settings
    client_type: str = "advanced"  # basic, advanced, batch, newsdata
    user_agent: str = "Lindela-NewsAPI-Crawler/1.0"
    timeout: int = 30

    # Cache settings
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour

    # Rate limiting
    max_requests: int = 100  # Default for Developer plan
    window_seconds: int = 86400  # 24 hours

    # Default search parameters
    default_language: str = "en"
    default_page_size: int = 100
    default_sort_by: str = "publishedAt"

    # Regional focus
    region_focus: List[str] = field(default_factory=lambda: ["Ethiopia", "Somalia", "Sudan", "South Sudan", "Kenya"])

    # Content processing
    extract_keywords: bool = True
    detect_locations: bool = True
    analyze_sentiment: bool = False

    # Output settings
    output_format: str = "json"
    output_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize derived settings."""
        # Use environment variable for API key if not provided
        if not self.api_key:
            self.api_key = os.environ.get("NEWSAPI_KEY")
            
        # Use environment variable for NewsData.io API key if not provided
        if not self.newsdata_api_key:
            self.newsdata_api_key = os.environ.get("NEWSDATA_API_KEY")

        # Set default cache directory if not provided
        if self.enable_caching and not self.cache_dir:
            self.cache_dir = os.path.join(os.path.expanduser("~"), ".newsapi_cache")
            os.makedirs(self.cache_dir, exist_ok=True)

        # Set default output directory if not provided
        if not self.output_dir:
            self.output_dir = os.path.join(os.path.expanduser("~"), "newsapi_results")
            os.makedirs(self.output_dir, exist_ok=True)

    def validate(self) -> bool:
        """
        Validate the configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if self.client_type == "newsdata":
            if not self.newsdata_api_key:
                raise ValueError("NewsData.io API key is required. Provide it directly or set NEWSDATA_API_KEY environment variable.")
        else:
            if not self.api_key:
                raise ValueError("API key is required. Provide it directly or set NEWSAPI_KEY environment variable.")

        if self.client_type not in ["basic", "advanced", "batch", "newsdata"]:
            raise ValueError(f"Invalid client type: {self.client_type}")

        if self.enable_caching and not self.cache_dir:
            raise ValueError("Cache directory is required when caching is enabled")

        if self.max_requests <= 0:
            raise ValueError(f"Invalid max_requests: {self.max_requests}")

        if self.window_seconds <= 0:
            raise ValueError(f"Invalid window_seconds: {self.window_seconds}")

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, file_path: str) -> bool:
        """
        Save configuration to a file.

        Args:
            file_path: Path to save the configuration

        Returns:
            True if successfully saved
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Configuration saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsAPIConfig':
        """
        Create configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            NewsAPIConfig object
        """
        return cls(**data)

    @classmethod
    def load(cls, file_path: str) -> 'NewsAPIConfig':
        """
        Load configuration from a file.

        Args:
            file_path: Path to the configuration file

        Returns:
            NewsAPIConfig object

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file isn't valid JSON
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        config = cls.from_dict(data)
        logger.info(f"Configuration loaded from {file_path}")
        return config


def get_default_config() -> NewsAPIConfig:
    """
    Get default configuration.

    Returns:
        Default NewsAPIConfig
    """
    return NewsAPIConfig()


def load_config_from_file(file_path: str) -> NewsAPIConfig:
    """
    Load configuration from a file.

    Args:
        file_path: Path to the configuration file

    Returns:
        NewsAPIConfig object
    """
    return NewsAPIConfig.load(file_path)


def save_config_to_file(config: NewsAPIConfig, file_path: str) -> bool:
    """
    Save configuration to a file.

    Args:
        config: Configuration to save
        file_path: Path to save the configuration

    Returns:
        True if successfully saved
    """
    return config.save(file_path)


def create_config_with_defaults(api_key: Optional[str] = None, **overrides) -> NewsAPIConfig:
    """
    Create configuration with defaults and optional overrides.

    Args:
        api_key: NewsAPI API key
        **overrides: Configuration overrides

    Returns:
        NewsAPIConfig object
    """
    config_dict = asdict(get_default_config())

    if api_key:
        config_dict['api_key'] = api_key

    # Apply overrides
    for key, value in overrides.items():
        if key in config_dict:
            config_dict[key] = value

    return NewsAPIConfig.from_dict(config_dict)
