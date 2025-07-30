#!/usr/bin/env python3
"""
NewsAPI Client Factory Module
============================

Factory functions for creating NewsAPI client instances with
various configurations.

This module provides factory functions to create:
- Basic NewsAPI clients
- Advanced clients with caching and rate limiting
- Batch clients for processing multiple queries

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional, Union

from .newsapi_client import NewsAPIClient, AdvancedNewsAPIClient
from .newsdata_client import NewsDataClient

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".newsapi_cache")
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_RATE_LIMIT = 100  # Default for Developer plan
DEFAULT_RATE_WINDOW = 86400  # 24 hours


def create_client(api_key: Optional[str] = None) -> NewsAPIClient:
    """
    Create a basic NewsAPI client.

    Args:
        api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.

    Returns:
        NewsAPIClient: A basic NewsAPI client
    """
    try:
        return NewsAPIClient(api_key=api_key)
    except Exception as e:
        logger.error(f"Error creating NewsAPI client: {str(e)}")
        raise


def create_client(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    max_requests: int = DEFAULT_RATE_LIMIT,
    window_seconds: int = DEFAULT_RATE_WINDOW
) -> NewsAPIClient:
    """
    Create a NewsAPI client with caching and rate limiting.

    Args:
        api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.
        cache_dir: Directory to store cache files. If None, caching is disabled.
        cache_ttl: Cache TTL in seconds (default: 1 hour)
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds

    Returns:
        NewsAPIClient: A NewsAPI client
    """
    try:
        return NewsAPIClient(
            api_key=api_key,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            max_requests=max_requests,
            window_seconds=window_seconds
        )
    except Exception as e:
        logger.error(f"Error creating advanced NewsAPI client: {str(e)}")
        raise


def create_advanced_client(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    max_requests: int = DEFAULT_RATE_LIMIT,
    window_seconds: int = DEFAULT_RATE_WINDOW
) -> AdvancedNewsAPIClient:
    """
    Create an advanced NewsAPI client with caching and rate limiting.
    
    Args:
        api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.
        cache_dir: Directory to store cache files. If None, caching is disabled.
        cache_ttl: Cache TTL in seconds (default: 1 hour)
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds
        
    Returns:
        AdvancedNewsAPIClient: An advanced NewsAPI client
    """
    try:
        return AdvancedNewsAPIClient(
            api_key=api_key,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            max_requests=max_requests,
            window_seconds=window_seconds
        )
    except Exception as e:
        logger.error(f"Error creating advanced NewsAPI client: {str(e)}")
        raise


class BatchClient:
    """Client for processing multiple queries in batch."""

    def __init__(self, client: AdvancedNewsAPIClient):
        """
        Initialize the batch client.

        Args:
            client: Advanced NewsAPI client
        """
        self.client = client

    async def process_queries(self, queries: list, **kwargs) -> Dict[str, list]:
        """
        Process multiple queries.

        Args:
            queries: List of queries to process
            **kwargs: Additional parameters for the search

        Returns:
            Dictionary mapping queries to search results
        """
        results = {}
        for query in queries:
            try:
                results[query] = await self.client.search_articles(query, **kwargs)
            except Exception as e:
                logger.error(f"Error processing query '{query}': {str(e)}")
                results[query] = []
        return results

    async def process_sources(self, sources: list, **kwargs) -> Dict[str, list]:
        """
        Process multiple sources.

        Args:
            sources: List of source IDs
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary mapping source IDs to lists of articles
        """
        return await self.client.get_articles_by_sources(sources, **kwargs)

    async def close(self):
        """Close the underlying client."""
        await self.client.close()


def create_batch_client(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    **kwargs
) -> BatchClient:
    """
    Create a batch client for processing multiple queries.

    Args:
        api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.
        cache_dir: Directory to store cache files. If None, caching is disabled.
        **kwargs: Additional parameters for the advanced client

    Returns:
        BatchClient: A batch client
    """
    try:
        advanced_client = create_advanced_client(
            api_key=api_key,
            cache_dir=cache_dir,
            **kwargs
        )
        return BatchClient(advanced_client)
    except Exception as e:
        logger.error(f"Error creating batch client: {str(e)}")
        raise


def create_newsdata_client(
    api_key: Optional[str] = None,
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    max_requests: int = DEFAULT_RATE_LIMIT,
    window_seconds: int = DEFAULT_RATE_WINDOW,
    credit_file: Optional[str] = None
) -> NewsDataClient:
    """
    Create a NewsData.io client with credit management.

    Args:
        api_key: NewsData.io API key. If not provided, tries to get from NEWSDATA_API_KEY env var.
        cache_dir: Directory to store cache files. If None, caching is disabled.
        cache_ttl: Cache TTL in seconds (default: 1 hour)
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds
        credit_file: File to store credit usage tracking

    Returns:
        NewsDataClient: A NewsData.io client
    """
    try:
        return NewsDataClient(
            api_key=api_key,
            cache_dir=cache_dir,
            cache_ttl=cache_ttl,
            max_requests=max_requests,
            window_seconds=window_seconds,
            credit_file=credit_file
        )
    except Exception as e:
        logger.error(f"Error creating NewsData.io client: {str(e)}")
        raise


def create_from_config(config: Dict[str, Any]) -> Union[NewsAPIClient, AdvancedNewsAPIClient, BatchClient, NewsDataClient]:
    """
    Create a client from a configuration dictionary.

    Args:
        config: Configuration dictionary with client settings

    Returns:
        A NewsAPI client based on the configuration
    """
    client_type = config.get("client_type", "basic")
    api_key = config.get("api_key")

    if client_type == "basic":
        return create_client(api_key=api_key)
    elif client_type == "advanced":
        return create_advanced_client(
            api_key=api_key,
            cache_dir=config.get("cache_dir", DEFAULT_CACHE_DIR),
            cache_ttl=config.get("cache_ttl", DEFAULT_CACHE_TTL),
            max_requests=config.get("max_requests", DEFAULT_RATE_LIMIT),
            window_seconds=config.get("window_seconds", DEFAULT_RATE_WINDOW)
        )
    elif client_type == "batch":
        return create_batch_client(
            api_key=api_key,
            cache_dir=config.get("cache_dir", DEFAULT_CACHE_DIR),
            cache_ttl=config.get("cache_ttl", DEFAULT_CACHE_TTL),
            max_requests=config.get("max_requests", DEFAULT_RATE_LIMIT),
            window_seconds=config.get("window_seconds", DEFAULT_RATE_WINDOW)
        )
    elif client_type == "newsdata":
        return create_newsdata_client(
            api_key=api_key,
            cache_dir=config.get("cache_dir", DEFAULT_CACHE_DIR),
            cache_ttl=config.get("cache_ttl", DEFAULT_CACHE_TTL),
            max_requests=config.get("max_requests", DEFAULT_RATE_LIMIT),
            window_seconds=config.get("window_seconds", DEFAULT_RATE_WINDOW),
            credit_file=config.get("credit_file")
        )
    else:
        raise ValueError(f"Unknown client type: {client_type}")
