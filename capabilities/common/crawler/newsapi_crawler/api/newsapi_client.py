#!/usr/bin/env python3
"""
NewsAPI Client Module
=====================

Advanced client implementation for the NewsAPI service with caching,
rate limiting, and enhanced functionality.

This module provides:
- Basic NewsAPIClient that wraps the official newsapi-python client
- Advanced NewsAPIAdvancedClient with caching and rate limiting
- Rate limiter to manage API quota efficiently

Author: Nyimbi Odero
Company: Datacraft (www.datacraft.co.ke)
License: MIT
"""

import os
import time
import json
import asyncio
import logging
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import aiohttp
from urllib.parse import urlencode

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# API Constants
BASE_URL = "https://newsapi.org/v2"
ENDPOINTS = {
    "top_headlines": "/top-headlines",
    "everything": "/everything",
    "sources": "/top-headlines/sources"
}
DEFAULT_PAGE_SIZE = 100
MAX_PAGE_SIZE = 100
RATE_LIMIT_REQUESTS = 100  # Default per day on Developer plan
RATE_LIMIT_WINDOW = 86400  # 24 hours in seconds


class NewsAPIRateLimiter:
    """Rate limiter for the NewsAPI service to manage API quota efficiently."""

    def __init__(self,
                 max_requests: int = RATE_LIMIT_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_timestamps = []
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """
        Acquire permission to make a request.

        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            # Remove timestamps outside the current window
            current_time = time.time()
            window_start = current_time - self.window_seconds
            self.request_timestamps = [t for t in self.request_timestamps if t >= window_start]

            # Check if we have capacity
            if len(self.request_timestamps) < self.max_requests:
                self.request_timestamps.append(current_time)
                return True
            return False

    async def wait_for_capacity(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until capacity is available to make a request.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if capacity became available, False if timed out
        """
        start_time = time.time()
        while timeout is None or time.time() - start_time < timeout:
            if await self.acquire():
                return True

            # Calculate time until next request slot opens
            async with self.lock:
                if not self.request_timestamps:
                    return True

                oldest_timestamp = min(self.request_timestamps)
                next_available = oldest_timestamp + self.window_seconds
                wait_time = max(0.1, min(next_available - time.time(), 1.0))

            await asyncio.sleep(wait_time)

        return False

    def get_remaining_requests(self) -> int:
        """
        Get the number of remaining requests in the current window.

        Returns:
            Number of remaining requests
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        active_requests = [t for t in self.request_timestamps if t >= window_start]
        return max(0, self.max_requests - len(active_requests))

    def get_reset_time(self) -> float:
        """
        Get the time when the rate limit will reset.

        Returns:
            Seconds until rate limit reset
        """
        if not self.request_timestamps:
            return 0

        oldest_timestamp = min(self.request_timestamps)
        return max(0, oldest_timestamp + self.window_seconds - time.time())


class NewsAPIClient:
    """
    Basic client for the NewsAPI service.

    This is a thin wrapper around the official newsapi-python client.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the NewsAPI client.

        Args:
            api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.
        """
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")

        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set NEWSAPI_KEY environment variable.")

        if not NEWSAPI_AVAILABLE:
            raise ImportError("newsapi-python package not available. Install it with pip install newsapi-python")

        self.client = NewsApiClient(api_key=self.api_key)
        logger.info("NewsAPI client initialized")

    def get_top_headlines(self, **kwargs) -> Dict[str, Any]:
        """
        Get top headlines from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - q: Keywords or phrases to search for in the article title and body
                - sources: A comma-separated string of identifiers for the news sources or blogs
                - category: The category to get headlines for
                - language: The 2-letter ISO-639-1 code of the language
                - country: The 2-letter ISO 3166-1 code of the country
                - page_size: The number of results to return per page (max: 100)
                - page: The page number to request

        Returns:
            Dictionary with the API response
        """
        return self.client.get_top_headlines(**kwargs)

    def get_everything(self, **kwargs) -> Dict[str, Any]:
        """
        Get all articles from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - q: Keywords or phrases to search for in the article title and body
                - sources: A comma-separated string of identifiers for the news sources or blogs
                - domains: A comma-separated string of domains
                - exclude_domains: A comma-separated string of domains to exclude
                - from_param: A date and optional time for the oldest article allowed
                - to: A date and optional time for the newest article allowed
                - language: The 2-letter ISO-639-1 code of the language
                - sort_by: The order to sort the articles in (relevancy, popularity, publishedAt)
                - page_size: The number of results to return per page (max: 100)
                - page: The page number to request

        Returns:
            Dictionary with the API response
        """
        return self.client.get_everything(**kwargs)

    def get_sources(self, **kwargs) -> Dict[str, Any]:
        """
        Get all available sources from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - category: Find sources that display news of this category
                - language: Find sources that display news in a specific language
                - country: Find sources that display news in a specific country

        Returns:
            Dictionary with the API response
        """
        return self.client.get_sources(**kwargs)


class NewsAPIClient:
    """
    Client for the NewsAPI service with caching, rate limiting,
    and functionality.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 cache_ttl: int = 3600,
                 max_requests: int = RATE_LIMIT_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW):
        """
        Initialize the advanced NewsAPI client.

        Args:
            api_key: NewsAPI API key. If not provided, tries to get from NEWSAPI_KEY env var.
            cache_dir: Directory to store cache files. If None, caching is disabled.
            cache_ttl: Cache TTL in seconds (default: 1 hour)
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
        """
        self.api_key = api_key or os.environ.get("NEWSAPI_KEY")

        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set NEWSAPI_KEY environment variable.")

        # Configure caching
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

        # Configure rate limiting
        self.rate_limiter = NewsAPIRateLimiter(max_requests, window_seconds)

        # Setup HTTP session
        self.timeout = aiohttp.ClientTimeout(total=30)
        self._session = None

        logger.info("NewsAPI advanced client initialized")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the NewsAPI.

        Args:
            endpoint: API endpoint to call
            params: Parameters for the API call

        Returns:
            Dictionary with the API response
        """
        # Check cache first
        cache_key = self._get_cache_key(endpoint, params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data

        # Wait for rate limit capacity
        if not await self.rate_limiter.wait_for_capacity(timeout=60):
            raise Exception("Rate limit exceeded and timed out waiting for capacity")

        # Make request
        url = f"{BASE_URL}{endpoint}"
        headers = {
            "X-Api-Key": self.api_key,
            "User-Agent": "NewsAPIAdvancedClient/1.0"
        }

        session = await self._get_session()

        try:
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json()

                if response.status != 200:
                    message = data.get("message", "Unknown error")
                    logger.error(f"API error: {message} (status: {response.status})")

                    if response.status == 429:
                        # Handle rate limiting
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        logger.warning(f"Rate limited. Retry after {retry_after} seconds")

                    raise Exception(f"NewsAPI error: {message}")

                # Cache successful response
                self._save_to_cache(cache_key, data)

                return data

        except aiohttp.ClientError as e:
            logger.error(f"HTTP error: {str(e)}")
            raise Exception(f"Network error: {str(e)}")

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the request."""
        # Create a stable representation of the parameters
        param_str = json.dumps(params, sort_keys=True)
        # Create a hash of the endpoint and parameters
        key = hashlib.md5(f"{endpoint}:{param_str}".encode()).hexdigest()
        return key

    def _get_cache_path(self, cache_key: str) -> str:
        """Get the file path for a cache key."""
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if available and not expired.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not available
        """
        if not self.cache_dir:
            return None

        cache_path = self._get_cache_path(cache_key)
        if not os.path.exists(cache_path):
            return None

        # Check if cache is expired
        file_modified_time = os.path.getmtime(cache_path)
        if time.time() - file_modified_time > self.cache_ttl:
            logger.debug(f"Cache expired for {cache_key}")
            return None

        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"Error reading cache: {str(e)}")
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> bool:
        """
        Save data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.cache_dir:
            return False

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            return True
        except Exception as e:
            logger.warning(f"Error writing to cache: {str(e)}")
            return False

    async def get_top_headlines(self, **kwargs) -> Dict[str, Any]:
        """
        Get top headlines from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - q: Keywords or phrases to search for in the article title and body
                - sources: A comma-separated string of identifiers for the news sources or blogs
                - category: The category to get headlines for
                - language: The 2-letter ISO-639-1 code of the language
                - country: The 2-letter ISO 3166-1 code of the country
                - page_size: The number of results to return per page (max: 100)
                - page: The page number to request

        Returns:
            Dictionary with the API response
        """
        endpoint = ENDPOINTS["top_headlines"]
        return await self._make_request(endpoint, kwargs)

    async def get_everything(self, **kwargs) -> Dict[str, Any]:
        """
        Get all articles from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - q: Keywords or phrases to search for in the article title and body
                - sources: A comma-separated string of identifiers for the news sources or blogs
                - domains: A comma-separated string of domains
                - exclude_domains: A comma-separated string of domains to exclude
                - from_param: A date and optional time for the oldest article allowed
                - to: A date and optional time for the newest article allowed
                - language: The 2-letter ISO-639-1 code of the language
                - sort_by: The order to sort the articles in (relevancy, popularity, publishedAt)
                - page_size: The number of results to return per page (max: 100)
                - page: The page number to request

        Returns:
            Dictionary with the API response
        """
        # Handle special parameters
        if "from_param" in kwargs:
            kwargs["from"] = kwargs.pop("from_param")

        endpoint = ENDPOINTS["everything"]
        return await self._make_request(endpoint, kwargs)

    async def get_sources(self, **kwargs) -> Dict[str, Any]:
        """
        Get all available sources from the NewsAPI.

        Args:
            **kwargs: Parameters for the API call
                - category: Find sources that display news of this category
                - language: Find sources that display news in a specific language
                - country: Find sources that display news in a specific country

        Returns:
            Dictionary with the API response
        """
        endpoint = ENDPOINTS["sources"]
        return await self._make_request(endpoint, kwargs)

    async def search_articles(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for articles with pagination handling.

        This method automatically handles pagination to get all results
        for the query up to the specified maximum.

        Args:
            query: Search query
            **kwargs: Additional parameters for the search
                - sources: A comma-separated string of identifiers for the news sources or blogs
                - domains: A comma-separated string of domains
                - from_param: A date and optional time for the oldest article allowed
                - to: A date and optional time for the newest article allowed
                - language: The 2-letter ISO-639-1 code of the language
                - sort_by: The order to sort the articles in (relevancy, popularity, publishedAt)
                - max_results: Maximum number of results to return (default: 100)

        Returns:
            List of articles matching the query
        """
        max_results = kwargs.pop("max_results", 100)
        page_size = min(MAX_PAGE_SIZE, max_results)

        params = {
            "q": query,
            "page_size": page_size,
            "page": 1,
            **kwargs
        }

        all_articles = []
        total_pages = 1
        current_page = 1

        while current_page <= total_pages and len(all_articles) < max_results:
            params["page"] = current_page

            response = await self.get_everything(**params)

            articles = response.get("articles", [])
            all_articles.extend(articles)

            total_results = response.get("totalResults", 0)
            total_pages = (total_results + page_size - 1) // page_size

            current_page += 1

            if current_page <= total_pages:
                # Small delay between requests to be nice to the API
                await asyncio.sleep(0.5)

        return all_articles[:max_results]

    async def get_articles_by_sources(self, sources: List[str], **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get articles from multiple sources.

        Args:
            sources: List of source IDs
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary mapping source IDs to lists of articles
        """
        results = {}

        for source in sources:
            params = {
                "sources": source,
                **kwargs
            }

            try:
                response = await self.get_everything(**params)
                results[source] = response.get("articles", [])
            except Exception as e:
                logger.error(f"Error getting articles for source {source}: {str(e)}")
                results[source] = []

        return results

    async def search_with_date_range(self, query: str, start_date: datetime,
                                     end_date: datetime, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for articles within a specific date range.

        Args:
            query: Search query
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters for the API call

        Returns:
            List of articles matching the query in the date range
        """
        # Format dates as ISO strings
        from_param = start_date.strftime("%Y-%m-%d")
        to = end_date.strftime("%Y-%m-%d")

        params = {
            "from_param": from_param,
            "to": to,
            **kwargs
        }

        return await self.search_articles(query, **params)

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        Get the current rate limit status.

        Returns:
            Dictionary with rate limit information
        """
        return {
            "remaining": self.rate_limiter.get_remaining_requests(),
            "reset_in_seconds": self.rate_limiter.get_reset_time(),
            "max_requests": self.rate_limiter.max_requests,
            "window_seconds": self.rate_limiter.window_seconds
        }

    def clear_cache(self) -> int:
        """
        Clear all cached responses.

        Returns:
            Number of cache files deleted
        """
        if not self.cache_dir or not os.path.exists(self.cache_dir):
            return 0

        count = 0
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(".json"):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(cache_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {filename}: {str(e)}")

        return count


# Create aliases for backward compatibility
# The second NewsAPIClient class is the advanced one with caching and rate limiting
AdvancedNewsAPIClient = NewsAPIClient
NewsAPIAdvancedClient = NewsAPIClient  # Backward compatibility
