"""
NewsAPI Crawler Package
=======================

A comprehensive crawler for the NewsAPI service with advanced filtering,
caching, and data processing capabilities. This package is designed to efficiently
retrieve and process news articles from the NewsAPI, providing robust support
for conflict monitoring in the Horn of Africa region.

Key Features:
- Comprehensive API client for all NewsAPI endpoints
- Intelligent rate limiting and quota management
- Caching and deduplication of requests
- Advanced filtering and relevance sorting
- Content enrichment with NLP processing
- Conflict event detection in news articles
- Batch processing and async operation

Components:
- api/: NewsAPI client and related functionality
- parsers/: Content parsers and extractors
- config/: Configuration management
- models/: Data models for news articles and related entities
- utils/: Utility functions and helpers
- tests/: Test suite for the package

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "Nyimbi Odero"
__license__ = "MIT"

# Import main components
try:
    from .api.newsapi_client import (
        NewsAPIClient,
        AdvancedNewsAPIClient,
        NewsAPIRateLimiter
    )
    
    from .api.newsdata_client import (
        NewsDataClient,
        CreditUsage,
        NewsDataRateLimiter
    )
    
    from .utils.location_optimizer import (
        LocationOptimizer,
        SearchLevel,
        create_location_optimizer,
        get_priority_locations
    )

    from .models.article import (
        NewsArticle,
        NewsSource,
        ArticleCollection,
        SearchParameters
    )

    from .parsers.content_parser import (
        ArticleParser,
        ContentExtractor,
        EventDetector
    )

    from .config.configuration import (
        NewsAPIConfig,
        get_default_config,
        load_config_from_file
    )

    from .utils.helpers import (
        date_to_string,
        filter_articles_by_keywords,
        extract_locations,
        calculate_relevance_score
    )

    # Factory functions for easy creation of client instances
    from .api.factory import (
        create_client,
        create_advanced_client,
        create_batch_client,
        create_newsdata_client
    )

    _COMPONENTS_AVAILABLE = True
    logger.info(f"NewsAPI Crawler v{__version__} initialized successfully")

except ImportError as e:
    logger.warning(f"Some components not available: {e}")
    _COMPONENTS_AVAILABLE = False

    # Define fallback classes for users
    class NewsAPIClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("NewsAPIClient not available")

    class AdvancedNewsAPIClient:
        def __init__(self, *args, **kwargs):
            raise ImportError("AdvancedNewsAPIClient not available")

    class ArticleCollection:
        def __init__(self, *args, **kwargs):
            raise ImportError("ArticleCollection not available")

# Convenience functions
async def search_news(query: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Quick function to search for news articles.

    Args:
        query: Search query
        **kwargs: Additional parameters for the search

    Returns:
        List of news articles matching the query
    """
    if not _COMPONENTS_AVAILABLE:
        raise ImportError("NewsAPI crawler components not available")

    client = create_client()
    return await client.search_articles(query, **kwargs)

async def get_top_headlines(query: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Quick function to get top headlines.

    Args:
        query: Optional search query
        **kwargs: Additional parameters

    Returns:
        List of top headline articles
    """
    if not _COMPONENTS_AVAILABLE:
        raise ImportError("NewsAPI crawler components not available")

    client = create_client()
    return await client.get_top_headlines(q=query, **kwargs)

def get_available_sources(**kwargs) -> List[Dict[str, Any]]:
    """
    Get available news sources from NewsAPI.

    Args:
        **kwargs: Filtering parameters

    Returns:
        List of available sources
    """
    if not _COMPONENTS_AVAILABLE:
        raise ImportError("NewsAPI crawler components not available")

    client = create_client()
    return client.get_sources(**kwargs)

def check_api_health() -> Dict[str, Any]:
    """
    Check the health of the NewsAPI integration.

    Returns:
        Dictionary with health information
    """
    status = {
        "version": __version__,
        "components_available": _COMPONENTS_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

    if _COMPONENTS_AVAILABLE:
        try:
            # Create client to check if API key is configured
            from .api.factory import create_client
            client = create_client()
            status["client_initialized"] = True

            # Check rate limit status if available
            if hasattr(client, 'get_rate_limit_status'):
                status["rate_limit"] = client.get_rate_limit_status()

            status["status"] = "healthy"
        except Exception as e:
            status["error"] = str(e)
            status["status"] = "degraded"
    else:
        status["status"] = "unavailable"

    return status

# Package exports
__all__ = [
    # Main classes
    "NewsAPIClient",
    "AdvancedNewsAPIClient",
    "NewsDataClient",
    "CreditUsage",
    "NewsArticle",
    "NewsSource",
    "ArticleCollection",
    "SearchParameters",

    # Factory functions
    "create_client",
    "create_advanced_client",
    "create_batch_client",
    "create_newsdata_client",

    # Convenience functions
    "search_news",
    "get_top_headlines",
    "get_available_sources",
    "check_api_health",

    # Configuration
    "NewsAPIConfig",
    "get_default_config",
    "load_config_from_file",

    # Utility functions
    "date_to_string",
    "filter_articles_by_keywords",
    "extract_locations",
    "calculate_relevance_score",

    # Version info
    "__version__",
    "__author__",
    "__license__"
]
