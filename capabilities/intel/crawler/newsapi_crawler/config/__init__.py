"""
NewsAPI Configuration Package
============================

This package contains configuration management for the NewsAPI crawler.

Components:
- configuration.py: Core configuration functionality

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

from .configuration import (
    NewsAPIConfig,
    get_default_config,
    load_config_from_file,
    save_config_to_file,
    create_config_with_defaults
)

__all__ = [
    "NewsAPIConfig",
    "get_default_config",
    "load_config_from_file",
    "save_config_to_file",
    "create_config_with_defaults"
]
