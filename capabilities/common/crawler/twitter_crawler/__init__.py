"""
Twitter Crawler Package
=======================

A comprehensive Twitter crawling and monitoring system using the twikit library.
Provides advanced Twitter data collection, analysis, and monitoring capabilities
specifically designed for conflict monitoring and social media intelligence.

This package offers:
- Authentication and session management with twikit
- Tweet search and scraping with advanced filters
- User profile and timeline crawling
- Trend and hashtag analysis
- Real-time monitoring capabilities
- Conflict-specific social media monitoring
- Rate limiting and error handling
- Data processing and storage integration

Author: Lindela Development Team
Version: 2.0.0
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime

# Version information
__version__ = "2.0.0"
__author__ = "Lindela Development Team"
__license__ = "MIT"

# Set up package logging
logger = logging.getLogger(__name__)

# Optional dependencies check
try:
    import twikit
    TWIKIT_AVAILABLE = True
except ImportError:
    TWIKIT_AVAILABLE = False
    logger.debug("twikit not available. Install with: pip install twikit")

try:
    import asyncio
    ASYNCIO_AVAILABLE = True
except ImportError:
    ASYNCIO_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Core imports with graceful fallbacks
try:
    from .core import (
        TwitterCrawler, TwitterConfig, TwitterSession,
        CrawlerError, AuthenticationError, RateLimitError
    )
    CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core Twitter crawler not available: {e}")
    CORE_AVAILABLE = False
    TwitterCrawler = None
    TwitterConfig = None
    TwitterSession = None
    CrawlerError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception

try:
    from .search import (
        TwitterSearchEngine, SearchQuery, SearchResult,
        TweetFilter, AdvancedSearchOptions
    )
    SEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Twitter search engine not available: {e}")
    SEARCH_AVAILABLE = False
    TwitterSearchEngine = None
    SearchQuery = None
    SearchResult = None
    TweetFilter = None
    AdvancedSearchOptions = None

try:
    from .monitoring import (
        TwitterMonitor, ConflictMonitor, TrendMonitor,
        AlertSystem, MonitoringConfig
    )
    MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Twitter monitoring not available: {e}")
    MONITORING_AVAILABLE = False

try:
    from .analysis import (
        TwitterAnalyzer, SentimentAnalyzer, TrendAnalyzer,
        NetworkAnalyzer, ConflictAnalyzer
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Twitter analysis not available: {e}")
    ANALYSIS_AVAILABLE = False

try:
    from .data import (
        TwitterDataProcessor, TweetModel, UserModel,
        DataExporter, DataStorage
    )
    DATA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Twitter data processing not available: {e}")
    DATA_AVAILABLE = False

# Create TwitterStreamer class (commonly expected for streaming functionality)
class TwitterStreamer:
    """Twitter streaming client for real-time tweet collection."""
    
    def __init__(self, config=None):
        if not TWIKIT_AVAILABLE:
            raise ImportError("twikit is required for TwitterStreamer. Install with: pip install twikit")
        self.config = config or {}
        self.is_streaming = False
        
    async def start_stream(self, filters=None, **kwargs):
        """Start streaming tweets with optional filters."""
        if not TWIKIT_AVAILABLE:
            raise ImportError("twikit not available")
        # Placeholder implementation
        self.is_streaming = True
        logger.info("Twitter stream started (placeholder implementation)")
        
    async def stop_stream(self):
        """Stop the current stream."""
        self.is_streaming = False
        logger.info("Twitter stream stopped")
        
    def is_active(self):
        """Check if stream is currently active."""
        return self.is_streaming

# Add TwitterResult class for compatibility
@dataclass
class TwitterResult:
    """Result object for Twitter operations."""
    success: bool = False
    data: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        from datetime import datetime
        if not hasattr(self, 'timestamp') or self.timestamp is None:
            self.timestamp = datetime.now()

# Quick access functions
def create_twitter_crawler(
    username: Optional[str] = None,
    password: Optional[str] = None,
    email: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
):
    """Create a configured Twitter crawler instance"""
    if not CORE_AVAILABLE:
        raise ImportError("Core Twitter crawler not available")
    
    # Create configuration with credentials
    config_dict = config or {}
    if username:
        config_dict['username'] = username
    if password:
        config_dict['password'] = password
    if email:
        config_dict['email'] = email
    
    crawler_config = TwitterConfig(**config_dict)
    crawler = TwitterCrawler(crawler_config)
    
    return crawler

def quick_search(
    query: str,
    max_results: int = 100,
    language: Optional[str] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Quick tweet search function"""
    if not SEARCH_AVAILABLE:
        raise ImportError("Twitter search engine not available")
    
    search_engine = TwitterSearchEngine()
    
    # Create TweetFilter with language and date filters
    tweet_filter = None
    if language or since_date or until_date:
        tweet_filter = TweetFilter(
            languages=[language] if language else [],
            since_date=since_date,
            until_date=until_date
        )
    
    search_query = SearchQuery(
        query=query,
        max_results=max_results,
        tweet_filter=tweet_filter
    )
    
    return search_engine.search(search_query)

def create_conflict_monitor(
    keywords: List[str],
    locations: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    alert_threshold: int = 10
):
    """Create a conflict monitoring system"""
    if not MONITORING_AVAILABLE:
        raise ImportError("Twitter monitoring not available")
    
    config = MonitoringConfig(
        keywords=keywords,
        locations=locations,
        languages=languages,
        alert_threshold=alert_threshold
    )
    
    return ConflictMonitor(config)

def get_package_info() -> Dict[str, Any]:
    """Get information about the Twitter crawler package"""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "dependencies": {
            "twikit": TWIKIT_AVAILABLE,
            "asyncio": ASYNCIO_AVAILABLE,
            "pandas": PANDAS_AVAILABLE,
            "numpy": NUMPY_AVAILABLE
        },
        "modules": {
            "core": CORE_AVAILABLE,
            "search": SEARCH_AVAILABLE,
            "monitoring": MONITORING_AVAILABLE,
            "analysis": ANALYSIS_AVAILABLE,
            "data": DATA_AVAILABLE
        }
    }

# Export main classes and functions
__all__ = [
    # Core components
    "TwitterCrawler", "TwitterConfig", "TwitterSession",
    "CrawlerError", "AuthenticationError", "RateLimitError",
    
    # Streaming components
    "TwitterStreamer",
    
    # Search components
    "TwitterSearchEngine", "SearchQuery", "SearchResult",
    "TweetFilter", "AdvancedSearchOptions",
    
    # Monitoring components
    "TwitterMonitor", "ConflictMonitor", "TrendMonitor",
    "AlertSystem", "MonitoringConfig",
    
    # Analysis components
    "TwitterAnalyzer", "SentimentAnalyzer", "TrendAnalyzer",
    "NetworkAnalyzer", "ConflictAnalyzer",
    
    # Data components
    "TwitterDataProcessor", "TweetModel", "UserModel",
    "DataExporter", "DataStorage", "TwitterResult",
    
    # Utility functions
    "create_twitter_crawler", "quick_search", "create_conflict_monitor",
    "get_package_info",
    
    # Constants
    "TWIKIT_AVAILABLE", "CORE_AVAILABLE", "SEARCH_AVAILABLE",
    "MONITORING_AVAILABLE", "ANALYSIS_AVAILABLE", "DATA_AVAILABLE"
]

# Package-level configuration
DEFAULT_CONFIG = {
    "rate_limit": {
        "requests_per_minute": 30,
        "requests_per_hour": 1000,
        "wait_on_rate_limit": True
    },
    "retry": {
        "max_retries": 3,
        "backoff_factor": 2,
        "retry_on_status": [429, 500, 502, 503, 504]
    },
    "timeout": {
        "connect_timeout": 30,
        "read_timeout": 60
    },
    "storage": {
        "enable_caching": True,
        "cache_ttl": 3600,
        "export_format": "json"
    },
    "monitoring": {
        "check_interval": 300,  # 5 minutes
        "alert_cooldown": 1800,  # 30 minutes
        "max_alerts_per_hour": 10
    }
}

def configure_package(config: Dict[str, Any]):
    """Configure package-wide settings"""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG.update(config)
    logger.info("Package configuration updated")

# Initialize package logging
def setup_logging(level: str = "INFO", format_string: Optional[str] = None):
    """Setup logging for the Twitter crawler package"""
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string
    )
    
    logger.info(f"Twitter crawler package initialized (v{__version__})")
    logger.info(f"twikit available: {TWIKIT_AVAILABLE}")

# Auto-setup logging on import
setup_logging()