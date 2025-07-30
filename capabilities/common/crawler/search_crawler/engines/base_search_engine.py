"""
Base Search Engine
==================

Abstract base class for search engine implementations.
Defines common interface and data structures.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Individual search result."""
    title: str
    url: str
    snippet: str
    engine: str
    rank: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed fields
    relevance_score: float = 0.0
    conflict_score: float = 0.0
    location_score: float = 0.0
    temporal_score: float = 0.0


@dataclass
class SearchResponse:
    """Complete search response from an engine."""
    query: str
    engine: str
    results: List[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSearchEngine(ABC):
    """Abstract base class for search engines."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Engine-specific configuration
        self.name = self.__class__.__name__.replace('SearchEngine', '').lower()
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_delay = self.config.get('min_delay', 2.0)
        self.max_delay = self.config.get('max_delay', 5.0)
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_results': 0,
            'average_response_time': 0.0
        }
    
    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 10,
        offset: int = 0,
        **kwargs
    ) -> SearchResponse:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            offset: Offset for pagination
            **kwargs: Engine-specific parameters
            
        Returns:
            SearchResponse object
        """
        pass
    
    @abstractmethod
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build the search URL for this engine."""
        pass
    
    @abstractmethod
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        """Parse HTML response to extract search results."""
        pass
    
    async def _make_request(self, url: str) -> Optional[str]:
        """Make HTTP request with rate limiting and error handling."""
        # Import here to avoid circular dependencies
        try:
            import aiohttp
            import random
        except ImportError:
            self.logger.error("aiohttp not available for search requests")
            return None
        
        # Rate limiting
        await self._rate_limit()
        
        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        return None
        
        except Exception as e:
            self.logger.error(f"Request failed for {url}: {e}")
            return None
    
    async def _rate_limit(self):
        """Implement rate limiting between requests."""
        import random
        import time
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        delay = random.uniform(self.min_delay, self.max_delay)
        
        if time_since_last < delay:
            sleep_time = delay - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _extract_clean_text(self, html_text: str) -> str:
        """Extract clean text from HTML."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_text, 'html.parser')
            return soup.get_text(strip=True)
        except ImportError:
            # Fallback: simple HTML tag removal
            import re
            return re.sub(r'<[^>]+>', '', html_text).strip()
    
    def _update_stats(self, success: bool, response_time: float, num_results: int):
        """Update engine statistics."""
        self.stats['total_searches'] += 1
        
        if success:
            self.stats['successful_searches'] += 1
            self.stats['total_results'] += num_results
        else:
            self.stats['failed_searches'] += 1
        
        # Update average response time
        total_successful = self.stats['successful_searches']
        if total_successful > 0:
            current_avg = self.stats['average_response_time']
            self.stats['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        success_rate = 0.0
        if self.stats['total_searches'] > 0:
            success_rate = self.stats['successful_searches'] / self.stats['total_searches']
        
        return {
            'engine_name': self.name,
            'total_searches': self.stats['total_searches'],
            'success_rate': success_rate,
            'total_results': self.stats['total_results'],
            'average_response_time': self.stats['average_response_time'],
            'average_results_per_search': (
                self.stats['total_results'] / max(self.stats['successful_searches'], 1)
            )
        }
    
    def reset_stats(self):
        """Reset engine statistics."""
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_results': 0,
            'average_response_time': 0.0
        }
    
    def is_healthy(self) -> bool:
        """Check if the engine is performing well."""
        if self.stats['total_searches'] < 5:
            return True  # Not enough data
        
        success_rate = self.stats['successful_searches'] / self.stats['total_searches']
        return success_rate >= 0.7  # 70% success rate threshold
    
    async def test_search(self) -> bool:
        """Test the search engine with a simple query."""
        try:
            response = await self.search("test", max_results=1)
            return response.success and len(response.results) > 0
        except Exception as e:
            self.logger.error(f"Test search failed: {e}")
            return False
    
    def __str__(self):
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self):
        return self.__str__()