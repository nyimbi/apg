#!/usr/bin/env python3
"""
NewsData.io Client Module
=========================

Client implementation for the NewsData.io API with credit management,
rate limiting, and integration with the news_crawler for stealth downloads.

This module provides:
- NewsDataClient for direct API access
- CreditManager for tracking API usage
- Integration with news_crawler for article content extraction
- Careful resource management for free plan limitations

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
from dataclasses import dataclass, asdict
import aiohttp
from urllib.parse import urlencode

# Import location optimizer
from ..utils.location_optimizer import LocationOptimizer, SearchLevel, create_location_optimizer

# Configure logging
logger = logging.getLogger(__name__)

# API Constants
BASE_URL = "https://newsdata.io/api/1"
ENDPOINTS = {
    "latest": "/latest",
    "archive": "/archive",
    "sources": "/sources",
    "crypto": "/crypto"
}
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50
FREE_PLAN_CREDITS = 200
ARTICLES_PER_CREDIT = 10
RATE_LIMIT_REQUESTS = 100  # Requests per day
RATE_LIMIT_WINDOW = 86400  # 24 hours in seconds


@dataclass
class CreditUsage:
    """Track credit usage for NewsData.io API."""
    total_credits: int = FREE_PLAN_CREDITS
    used_credits: int = 0
    remaining_credits: int = FREE_PLAN_CREDITS
    last_updated: datetime = None
    daily_usage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.daily_usage is None:
            self.daily_usage = {}
    
    def update_usage(self, articles_fetched: int):
        """Update credit usage based on articles fetched."""
        credits_used = max(1, (articles_fetched + ARTICLES_PER_CREDIT - 1) // ARTICLES_PER_CREDIT)
        self.used_credits += credits_used
        self.remaining_credits = max(0, self.total_credits - self.used_credits)
        self.last_updated = datetime.now()
        
        # Track daily usage
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_usage:
            self.daily_usage[today] = 0
        self.daily_usage[today] += credits_used
        
        logger.info(f"Credit usage updated: {credits_used} credits used, {self.remaining_credits} remaining")
    
    def can_make_request(self, expected_articles: int = 10) -> bool:
        """Check if we have enough credits for the request."""
        expected_credits = max(1, (expected_articles + ARTICLES_PER_CREDIT - 1) // ARTICLES_PER_CREDIT)
        return self.remaining_credits >= expected_credits
    
    def get_recommended_page_size(self, max_articles: int = None) -> int:
        """Get recommended page size to optimize credit usage."""
        if max_articles is None:
            max_articles = self.remaining_credits * ARTICLES_PER_CREDIT
        
        # Use smaller page sizes to be more conservative
        return min(DEFAULT_PAGE_SIZE, max_articles, self.remaining_credits * ARTICLES_PER_CREDIT)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_credits": self.total_credits,
            "used_credits": self.used_credits,
            "remaining_credits": self.remaining_credits,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "daily_usage": self.daily_usage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreditUsage':
        """Create from dictionary."""
        credit_usage = cls()
        credit_usage.total_credits = data.get("total_credits", FREE_PLAN_CREDITS)
        credit_usage.used_credits = data.get("used_credits", 0)
        credit_usage.remaining_credits = data.get("remaining_credits", FREE_PLAN_CREDITS)
        
        last_updated_str = data.get("last_updated")
        if last_updated_str:
            credit_usage.last_updated = datetime.fromisoformat(last_updated_str)
        
        credit_usage.daily_usage = data.get("daily_usage", {})
        return credit_usage


class NewsDataRateLimiter:
    """Rate limiter for the NewsData.io API."""

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


class NewsDataClient:
    """
    Client for the NewsData.io API with credit management and rate limiting.
    
    This client is designed for efficient use of the free plan's 200 credits,
    with automatic credit tracking and optimization.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 cache_ttl: int = 3600,
                 max_requests: int = RATE_LIMIT_REQUESTS,
                 window_seconds: int = RATE_LIMIT_WINDOW,
                 credit_file: Optional[str] = None):
        """
        Initialize the NewsData.io client.

        Args:
            api_key: NewsData.io API key. If not provided, tries to get from NEWSDATA_API_KEY env var.
            cache_dir: Directory to store cache files. If None, caching is disabled.
            cache_ttl: Cache TTL in seconds (default: 1 hour)
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds
            credit_file: File to store credit usage tracking
        """
        self.api_key = api_key or os.environ.get("NEWSDATA_API_KEY")

        if not self.api_key:
            raise ValueError("API key is required. Provide it directly or set NEWSDATA_API_KEY environment variable.")

        # Configure caching
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logger.info(f"Created cache directory: {cache_dir}")

        # Configure rate limiting
        self.rate_limiter = NewsDataRateLimiter(max_requests, window_seconds)

        # Setup HTTP session
        self.timeout = aiohttp.ClientTimeout(total=30)
        self._session = None

        # Credit management
        self.credit_file = credit_file or os.path.join(
            cache_dir or os.path.expanduser("~"), 
            ".newsdata_credits.json"
        )
        self.credit_usage = self._load_credit_usage()

        # News crawler integration (will be set up later)
        self._news_crawler = None
        
        # Location optimizer for targeted searches
        self.location_optimizer = create_location_optimizer()

        logger.info("NewsData.io client initialized")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self):
        """Close the HTTP session and save credit usage."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
        self._save_credit_usage()

    def _load_credit_usage(self) -> CreditUsage:
        """Load credit usage from file."""
        if os.path.exists(self.credit_file):
            try:
                with open(self.credit_file, 'r') as f:
                    data = json.load(f)
                return CreditUsage.from_dict(data)
            except Exception as e:
                logger.warning(f"Error loading credit usage: {str(e)}")
        
        return CreditUsage()

    def _save_credit_usage(self):
        """Save credit usage to file."""
        try:
            os.makedirs(os.path.dirname(self.credit_file), exist_ok=True)
            with open(self.credit_file, 'w') as f:
                json.dump(self.credit_usage.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving credit usage: {str(e)}")

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the NewsData.io API.

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

        # Check credit availability
        expected_page_size = params.get('size', DEFAULT_PAGE_SIZE)
        if not self.credit_usage.can_make_request(expected_page_size):
            raise Exception(f"Insufficient credits. Remaining: {self.credit_usage.remaining_credits}")

        # Wait for rate limit capacity
        if not await self.rate_limiter.wait_for_capacity(timeout=60):
            raise Exception("Rate limit exceeded and timed out waiting for capacity")

        # Make request
        url = f"{BASE_URL}{endpoint}"
        headers = {
            "X-ACCESS-KEY": self.api_key,
            "User-Agent": "Lindela-NewsData-Client/1.0"
        }

        session = await self._get_session()

        try:
            async with session.get(url, params=params, headers=headers) as response:
                data = await response.json()

                if response.status != 200:
                    error_message = data.get("message", "Unknown error")
                    logger.error(f"API error: {error_message} (status: {response.status})")

                    if response.status == 429:
                        # Handle rate limiting
                        retry_after = int(response.headers.get("Retry-After", "60"))
                        logger.warning(f"Rate limited. Retry after {retry_after} seconds")

                    raise Exception(f"NewsData.io API error: {error_message}")

                # Update credit usage based on actual results
                results = data.get("results", [])
                if results:
                    self.credit_usage.update_usage(len(results))
                    self._save_credit_usage()

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
        return os.path.join(self.cache_dir, f"newsdata_{cache_key}.json")

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

    async def get_latest_news(self, **kwargs) -> Dict[str, Any]:
        """
        Get latest news articles.

        Args:
            **kwargs: Parameters for the API call
                - q: Query string for searching news
                - country: 2-letter ISO code for country-specific news
                - category: Category of news (business, entertainment, etc.)
                - language: 2-letter ISO code for language
                - domain: Domain to search within
                - timeframe: Time frame for news (1h, 24h, 7d, 30d)
                - size: Number of articles to return (max 50)

        Returns:
            Dictionary with the API response
        """
        # Optimize page size based on remaining credits
        if "size" not in kwargs:
            kwargs["size"] = self.credit_usage.get_recommended_page_size()
        else:
            kwargs["size"] = min(kwargs["size"], self.credit_usage.get_recommended_page_size())

        endpoint = ENDPOINTS["latest"]
        return await self._make_request(endpoint, kwargs)

    async def get_archive_news(self, **kwargs) -> Dict[str, Any]:
        """
        Get archived news articles.

        Args:
            **kwargs: Parameters for the API call
                - q: Query string for searching news
                - from_date: Start date (YYYY-MM-DD format)
                - to_date: End date (YYYY-MM-DD format)
                - country: 2-letter ISO code for country-specific news
                - category: Category of news
                - language: 2-letter ISO code for language
                - size: Number of articles to return (max 50)

        Returns:
            Dictionary with the API response
        """
        # Optimize page size based on remaining credits
        if "size" not in kwargs:
            kwargs["size"] = self.credit_usage.get_recommended_page_size()
        else:
            kwargs["size"] = min(kwargs["size"], self.credit_usage.get_recommended_page_size())

        endpoint = ENDPOINTS["archive"]
        return await self._make_request(endpoint, kwargs)

    async def get_sources(self, **kwargs) -> Dict[str, Any]:
        """
        Get available news sources.

        Args:
            **kwargs: Parameters for the API call
                - country: 2-letter ISO code for country-specific sources
                - category: Category of sources
                - language: 2-letter ISO code for language

        Returns:
            Dictionary with the API response
        """
        endpoint = ENDPOINTS["sources"]
        return await self._make_request(endpoint, kwargs)

    async def get_crypto_news(self, **kwargs) -> Dict[str, Any]:
        """
        Get cryptocurrency news.

        Args:
            **kwargs: Parameters for the API call
                - q: Query string for searching crypto news
                - coin: Specific cryptocurrency symbol
                - size: Number of articles to return (max 50)

        Returns:
            Dictionary with the API response
        """
        # Optimize page size based on remaining credits
        if "size" not in kwargs:
            kwargs["size"] = self.credit_usage.get_recommended_page_size()
        else:
            kwargs["size"] = min(kwargs["size"], self.credit_usage.get_recommended_page_size())

        endpoint = ENDPOINTS["crypto"]
        return await self._make_request(endpoint, kwargs)

    async def search_with_stealth_download(self, query: str, max_articles: int = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for news and download full article content using stealth crawler.
        
        This method fetches article metadata from NewsData.io, then uses the news_crawler
        to download the full content of each article.

        Args:
            query: Search query
            max_articles: Maximum number of articles to process
            **kwargs: Additional parameters for the search

        Returns:
            List of articles with full content
        """
        # Import news_crawler here to avoid circular imports
        try:
            from ...news_crawler import NewsCrawler
            if self._news_crawler is None:
                self._news_crawler = NewsCrawler()
        except ImportError as e:
            logger.error(f"Could not import news_crawler: {str(e)}")
            raise Exception("news_crawler not available for stealth downloads")

        # Determine how many articles to fetch based on credits
        if max_articles is None:
            max_articles = min(50, self.credit_usage.remaining_credits * ARTICLES_PER_CREDIT)

        # Search for articles
        params = {
            "q": query,
            "size": min(max_articles, self.credit_usage.get_recommended_page_size()),
            **kwargs
        }

        logger.info(f"Searching for news with query: '{query}' (max {max_articles} articles)")
        response = await self.get_latest_news(**params)

        articles = response.get("results", [])
        enhanced_articles = []

        logger.info(f"Found {len(articles)} articles, downloading full content...")

        for i, article in enumerate(articles):
            try:
                # Get article URL
                article_url = article.get("link")
                if not article_url:
                    logger.warning(f"No URL found for article {i+1}")
                    enhanced_articles.append(article)
                    continue

                # Download full content using stealth crawler
                logger.debug(f"Downloading content for: {article_url}")
                content_result = await self._news_crawler.download_article(article_url)

                if content_result and content_result.get("success"):
                    # Merge NewsData.io metadata with full content
                    enhanced_article = {
                        **article,  # Original NewsData.io data
                        "full_content": content_result.get("content", ""),
                        "extracted_text": content_result.get("text", ""),
                        "download_timestamp": datetime.now().isoformat(),
                        "download_success": True
                    }
                else:
                    # Keep original article if download failed
                    enhanced_article = {
                        **article,
                        "download_success": False,
                        "download_error": content_result.get("error") if content_result else "Unknown error"
                    }

                enhanced_articles.append(enhanced_article)

                # Small delay between downloads to be respectful
                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error processing article {i+1}: {str(e)}")
                enhanced_articles.append({
                    **article,
                    "download_success": False,
                    "download_error": str(e)
                })

        logger.info(f"Successfully processed {len(enhanced_articles)} articles")
        return enhanced_articles

    def get_credit_status(self) -> Dict[str, Any]:
        """
        Get the current credit usage status.

        Returns:
            Dictionary with credit information
        """
        return {
            "total_credits": self.credit_usage.total_credits,
            "used_credits": self.credit_usage.used_credits,
            "remaining_credits": self.credit_usage.remaining_credits,
            "remaining_articles": self.credit_usage.remaining_credits * ARTICLES_PER_CREDIT,
            "last_updated": self.credit_usage.last_updated.isoformat() if self.credit_usage.last_updated else None,
            "daily_usage": self.credit_usage.daily_usage,
            "recommended_page_size": self.credit_usage.get_recommended_page_size()
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
            if filename.startswith("newsdata_") and filename.endswith(".json"):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(cache_path)
                    count += 1
                except Exception as e:
                    logger.warning(f"Error deleting cache file {filename}: {str(e)}")

        return count

    def reset_credits(self, new_total: int = FREE_PLAN_CREDITS):
        """
        Reset credit usage (for testing or plan upgrades).

        Args:
            new_total: New total credit limit
        """
        self.credit_usage = CreditUsage(total_credits=new_total)
        self._save_credit_usage()
        logger.info(f"Credits reset to {new_total}")

    async def search_priority_locations(self, 
                                       locations: List[str] = None,
                                       max_credits: int = 10,
                                       include_stealth_download: bool = True) -> Dict[str, Any]:
        """
        Search for conflict news in priority locations using hierarchical strategy.
        
        This method implements the optimized search strategy:
        1. Start with specific locations (Aweil, Karamoja, Mandera, Assosa)
        2. Expand to district/county level if insufficient results
        3. Expand to regional/provincial level if still insufficient
        4. Search at country level as last resort

        Args:
            locations: List of location keys to search (default: all priority locations)
            max_credits: Maximum credits to use across all searches
            include_stealth_download: Whether to download full article content

        Returns:
            Dictionary with results by location and search statistics
        """
        if locations is None:
            locations = ["aweil", "karamoja", "mandera", "assosa"]

        # Check if we have enough credits
        if not self.credit_usage.can_make_request(max_credits * 10):  # Estimate 10 articles per credit
            raise Exception(f"Insufficient credits for priority location search. Need ~{max_credits}, have {self.credit_usage.remaining_credits}")

        logger.info(f"Starting priority location search for: {', '.join(locations)}")
        logger.info(f"Credit budget: {max_credits} credits")

        # Generate optimized search plan
        search_plan = self.location_optimizer.get_optimized_search_plan(locations, max_credits)
        
        results = {}
        total_credits_used = 0
        
        for location_plan in search_plan:
            location_key = location_plan["location"]
            location_data = location_plan["location_data"]
            
            logger.info(f"\n--- Searching {location_key.upper()} ---")
            
            location_results = {
                "articles": [],
                "credits_used": 0,
                "search_levels_tried": [],
                "final_level": None,
                "total_articles_found": 0
            }
            
            # Try hierarchical search levels
            for level in SearchLevel:
                if total_credits_used >= max_credits:
                    logger.warning(f"Credit budget exhausted, stopping search")
                    break
                    
                location_terms = location_data.get_search_terms(level)
                if not location_terms:
                    continue
                    
                logger.info(f"Trying {level.value} level: {location_terms}")
                location_results["search_levels_tried"].append(level.value)
                
                # Search for conflict news at this level
                level_articles = []
                for term in location_terms[:2]:  # Limit to 2 terms per level to save credits
                    if total_credits_used >= max_credits:
                        break
                        
                    # Create conflict-focused query
                    conflict_queries = [
                        f'"{term}" conflict',
                        f'"{term}" violence',
                        f'"{term}" attack',
                        f'"{term}" killed'
                    ]
                    
                    for query in conflict_queries[:2]:  # Try top 2 conflict terms
                        if total_credits_used >= max_credits:
                            break
                            
                        try:
                            logger.debug(f"Searching: {query}")
                            response = await self.get_latest_news(
                                q=query,
                                language="en",
                                size=min(5, self.credit_usage.get_recommended_page_size())
                            )
                            
                            articles = response.get("results", [])
                            if articles:
                                # Filter for relevance
                                relevant_articles = self.location_optimizer.filter_relevant_articles(
                                    articles, location_terms
                                )
                                level_articles.extend(relevant_articles)
                                
                            total_credits_used += 1
                            location_results["credits_used"] += 1
                            
                            # Small delay between requests
                            await asyncio.sleep(0.5)
                            
                        except Exception as e:
                            logger.error(f"Error searching '{query}': {str(e)}")
                            continue
                
                # Remove duplicates by URL
                seen_urls = set()
                unique_articles = []
                for article in level_articles:
                    url = article.get("link", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_articles.append(article)
                
                location_results["articles"].extend(unique_articles)
                location_results["total_articles_found"] = len(location_results["articles"])
                location_results["final_level"] = level.value
                
                logger.info(f"Found {len(unique_articles)} relevant articles at {level.value} level")
                
                # Check if we should continue to broader search
                if not self.location_optimizer.should_expand_search(unique_articles, level):
                    logger.info(f"Sufficient articles found for {location_key}, stopping search")
                    break
            
            # Download full content if requested
            if include_stealth_download and location_results["articles"]:
                logger.info(f"Downloading full content for {len(location_results['articles'])} articles...")
                enhanced_articles = []
                
                for article in location_results["articles"][:5]:  # Limit to 5 for performance
                    try:
                        if self._news_crawler is None:
                            from ...news_crawler import NewsCrawler
                            self._news_crawler = NewsCrawler()
                        
                        article_url = article.get("link")
                        if article_url:
                            content_result = await self._news_crawler.download_article(article_url)
                            if content_result and content_result.get("success"):
                                article.update({
                                    "full_content": content_result.get("content", ""),
                                    "extracted_text": content_result.get("text", ""),
                                    "download_success": True
                                })
                            else:
                                article["download_success"] = False
                        
                        enhanced_articles.append(article)
                        await asyncio.sleep(0.5)  # Be respectful
                        
                    except Exception as e:
                        logger.warning(f"Failed to download content: {str(e)}")
                        article["download_success"] = False
                        enhanced_articles.append(article)
                
                location_results["articles"] = enhanced_articles
            
            results[location_key] = location_results
            logger.info(f"Completed {location_key}: {location_results['total_articles_found']} articles, {location_results['credits_used']} credits")
        
        # Generate summary
        summary = self.location_optimizer.format_search_summary(results)
        logger.info(f"\n{summary}")
        
        # Add overall statistics
        results["_summary"] = {
            "total_credits_used": total_credits_used,
            "total_articles": sum(loc["total_articles_found"] for loc in results.values() if isinstance(loc, dict) and "total_articles_found" in loc),
            "locations_searched": len(locations),
            "credits_remaining": self.credit_usage.remaining_credits,
            "search_timestamp": datetime.now().isoformat()
        }
        
        return results

    async def monitor_location(self, 
                              location: str,
                              hours_back: int = 24,
                              max_credits: int = 3) -> Dict[str, Any]:
        """
        Monitor a specific priority location for recent conflict news.

        Args:
            location: Location key ('aweil', 'karamoja', 'mandera', 'assosa')
            hours_back: How many hours back to search
            max_credits: Maximum credits to use

        Returns:
            Dictionary with monitoring results
        """
        if location.lower() not in self.location_optimizer.priority_locations:
            raise ValueError(f"Unknown location: {location}. Available: {list(self.location_optimizer.priority_locations.keys())}")

        logger.info(f"Monitoring {location.upper()} for last {hours_back} hours")

        # Generate targeted queries
        queries = self.location_optimizer.generate_search_queries(location, conflict_focus=True)
        
        # Select top queries within credit budget
        selected_queries = queries[:max_credits]
        
        all_articles = []
        credits_used = 0
        
        for query_config in selected_queries:
            if credits_used >= max_credits:
                break
                
            try:
                query = query_config["query"]
                logger.debug(f"Monitoring query: {query}")
                
                response = await self.get_latest_news(
                    q=query,
                    language="en",
                    size=min(10, self.credit_usage.get_recommended_page_size())
                )
                
                articles = response.get("results", [])
                
                # Filter for recent articles
                recent_articles = []
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                
                for article in articles:
                    pub_date_str = article.get("pubDate", "")
                    try:
                        pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                        if pub_date >= cutoff_time:
                            recent_articles.append(article)
                    except:
                        # If date parsing fails, include the article
                        recent_articles.append(article)
                
                all_articles.extend(recent_articles)
                credits_used += 1
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in monitoring query: {str(e)}")
                continue
        
        # Remove duplicates and filter for relevance
        location_data = self.location_optimizer.priority_locations[location.lower()]
        all_location_terms = []
        for level in SearchLevel:
            all_location_terms.extend(location_data.get_search_terms(level))
        
        relevant_articles = self.location_optimizer.filter_relevant_articles(all_articles, all_location_terms)
        
        # Remove URL duplicates
        seen_urls = set()
        unique_articles = []
        for article in relevant_articles:
            url = article.get("link", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        results = {
            "location": location,
            "monitoring_period_hours": hours_back,
            "articles_found": len(unique_articles),
            "articles": unique_articles,
            "credits_used": credits_used,
            "queries_tried": len(selected_queries),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Monitoring complete: {len(unique_articles)} relevant articles found")
        return results

    async def alert_scan(self, 
                        alert_keywords: List[str] = None,
                        max_credits: int = 5) -> Dict[str, Any]:
        """
        Perform rapid alert scan across all priority locations for critical incidents.

        Args:
            alert_keywords: High-priority keywords to scan for (default: critical conflict terms)
            max_credits: Maximum credits to use for alert scan

        Returns:
            Dictionary with alert scan results
        """
        if alert_keywords is None:
            alert_keywords = ["killed", "attack", "violence", "raid", "clash", "explosion", "bombing"]

        logger.info(f"Performing alert scan with keywords: {', '.join(alert_keywords)}")

        all_locations = list(self.location_optimizer.priority_locations.keys())
        alerts = []
        credits_used = 0
        
        for location in all_locations:
            if credits_used >= max_credits:
                break
                
            location_data = self.location_optimizer.priority_locations[location]
            location_name = location_data.specific_location
            
            # Try most urgent combinations
            for keyword in alert_keywords[:3]:  # Top 3 alert keywords
                if credits_used >= max_credits:
                    break
                    
                query = f'"{location_name}" {keyword}'
                
                try:
                    response = await self.get_latest_news(
                        q=query,
                        language="en",
                        size=3  # Small size for rapid scanning
                    )
                    
                    articles = response.get("results", [])
                    
                    # Check for very recent articles (last 6 hours)
                    cutoff_time = datetime.now() - timedelta(hours=6)
                    
                    for article in articles:
                        pub_date_str = article.get("pubDate", "")
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                            if pub_date >= cutoff_time:
                                alerts.append({
                                    "location": location,
                                    "keyword": keyword,
                                    "article": article,
                                    "urgency": "high" if keyword in ["killed", "attack", "explosion"] else "medium",
                                    "hours_ago": (datetime.now() - pub_date).total_seconds() / 3600
                                })
                        except:
                            # If recent, include anyway
                            alerts.append({
                                "location": location,
                                "keyword": keyword,
                                "article": article,
                                "urgency": "medium",
                                "hours_ago": "unknown"
                            })
                    
                    credits_used += 1
                    await asyncio.sleep(0.3)  # Faster for alerts
                    
                except Exception as e:
                    logger.error(f"Error in alert scan for {location} + {keyword}: {str(e)}")
                    continue
        
        # Sort alerts by urgency and recency
        alerts.sort(key=lambda x: (x["urgency"] == "high", -x.get("hours_ago", 999) if isinstance(x.get("hours_ago"), (int, float)) else 0), reverse=True)
        
        result = {
            "alert_count": len(alerts),
            "high_urgency_count": len([a for a in alerts if a["urgency"] == "high"]),
            "alerts": alerts,
            "credits_used": credits_used,
            "scan_timestamp": datetime.now().isoformat(),
            "locations_scanned": len(all_locations)
        }
        
        if alerts:
            logger.warning(f"⚠️  ALERTS FOUND: {len(alerts)} potential incidents across priority locations")
        else:
            logger.info("✅ No urgent alerts found in priority locations")
            
        return result