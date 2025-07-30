"""
Google News Crawler CLI Package
==============================

Command-line interface for the Google News Crawler with support for searching,
configuration management, monitoring, and Crawlee integration.

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
Date: June 28, 2025
"""

__version__ = "1.0.0"

from .main import main_cli
from .commands import (
    search_command,
    config_command,
    monitor_command,
    crawlee_command,
    status_command
)

__all__ = [
    'main_cli',
    'search_command',
    'config_command', 
    'monitor_command',
    'crawlee_command',
    'status_command'
]