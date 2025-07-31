"""
Google Search Engine
====================

Google search implementation with result parsing and stealth capabilities.

Author: Lindela Development Team
Version: 1.0.0
License: MIT
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote, urlparse
from datetime import datetime

from .base_search_engine import BaseSearchEngine, SearchResult, SearchResponse

logger = logging.getLogger(__name__)


class GoogleSearchEngine(BaseSearchEngine):
    """Google search engine implementation."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Google-specific configuration
        self.base_url = "https://www.google.com/search"
        self.min_delay = self.config.get('min_delay', 3.0)  # Google is strict
        self.max_delay = self.config.get('max_delay', 7.0)
        
        # Additional headers for Google
        self.google_headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
        
        # Google-specific selectors
        self.result_selectors = {
            'container': 'div.g',
            'title': 'h3',
            'link': 'a[href]',
            'snippet': 'span[data-ved] span, .VwiC3b'
        }
    
    def _build_search_url(self, query: str, offset: int = 0, **kwargs) -> str:
        """Build Google search URL."""
        params = {
            'q': query,
            'start': offset,
            'num': kwargs.get('num', 10),
            'hl': kwargs.get('language', 'en'),
            'gl': kwargs.get('country', 'us'),
            'safe': kwargs.get('safe', 'off')
        }
        
        # Add time filter if specified
        if 'time_filter' in kwargs:
            time_filters = {
                'day': 'd',
                'week': 'w', 
                'month': 'm',
                'year': 'y'
            }
            if kwargs['time_filter'] in time_filters:
                params['tbs'] = f"qdr:{time_filters[kwargs['time_filter']]}"
        
        # Build URL
        param_string = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
        return f"{self.base_url}?{param_string}"
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        offset: int = 0,
        **kwargs
    ) -> SearchResponse:
        """Perform Google search."""
        start_time = time.time()
        
        try:
            # Build search URL
            search_url = self._build_search_url(query, offset, num=max_results, **kwargs)
            self.logger.debug(f"Google search URL: {search_url}")
            
            # Make request
            html = await self._make_request_with_google_headers(search_url)
            if not html:
                return SearchResponse(
                    query=query,
                    engine=self.name,
                    success=False,
                    error_message="Failed to fetch search results"
                )
            
            # Parse results
            results = await self._parse_results(html, query)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(True, response_time, len(results))
            
            return SearchResponse(
                query=query,
                engine=self.name,
                results=results,
                total_results=len(results),
                search_time=response_time,
                success=True
            )
        
        except Exception as e:
            self.logger.error(f"Google search failed for query '{query}': {e}")
            response_time = time.time() - start_time
            self._update_stats(False, response_time, 0)
            
            return SearchResponse(
                query=query,
                engine=self.name,
                success=False,
                error_message=str(e),
                search_time=response_time
            )
    
    async def _make_request_with_google_headers(self, url: str) -> Optional[str]:
        """Make request with Google-specific headers."""
        try:
            import aiohttp
            import random
        except ImportError:
            self.logger.error("aiohttp not available")
            return None
        
        # Rate limiting
        await self._rate_limit()
        
        # Combine headers
        headers = {
            'User-Agent': random.choice(self.user_agents),
            **self.google_headers
        }
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:
                        self.logger.warning("Google rate limit hit")
                        # Exponential backoff
                        await asyncio.sleep(random.uniform(10, 20))
                        return None
                    else:
                        self.logger.warning(f"Google returned HTTP {response.status}")
                        return None
        
        except Exception as e:
            self.logger.error(f"Google request failed: {e}")
            return None
    
    async def _parse_results(self, html: str, query: str) -> List[SearchResult]:
        """Parse Google search results from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            self.logger.error("BeautifulSoup not available for parsing")
            return []
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find result containers
            result_containers = soup.find_all('div', class_='g')
            
            for i, container in enumerate(result_containers):
                try:
                    # Extract title and link
                    title_element = container.find('h3')
                    if not title_element:
                        continue
                    
                    title = title_element.get_text(strip=True)
                    
                    # Find link (usually in parent anchor)
                    link_element = title_element.find_parent('a')
                    if not link_element:
                        link_element = container.find('a')
                    
                    if not link_element or not link_element.get('href'):
                        continue
                    
                    url = link_element['href']
                    
                    # Clean Google redirect URLs
                    if url.startswith('/url?'):
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(url[5:])
                        if 'url' in parsed:
                            url = parsed['url'][0]
                    
                    # Extract snippet
                    snippet = ""
                    snippet_selectors = [
                        'span[data-ved]',
                        '.VwiC3b',
                        '.s3v9rd',
                        '.hgKElc'
                    ]
                    
                    for selector in snippet_selectors:
                        snippet_element = container.select_one(selector)
                        if snippet_element:
                            snippet = snippet_element.get_text(strip=True)
                            break
                    
                    if not snippet:
                        # Fallback: get any text from container
                        snippet = container.get_text(strip=True)[:200]
                    
                    # Create search result
                    result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        engine=self.name,
                        rank=i + 1,
                        metadata={'source': 'google'}
                    )
                    
                    results.append(result)
                
                except Exception as e:
                    self.logger.debug(f"Error parsing Google result {i}: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(results)} Google results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error parsing Google results: {e}")
            return []
    
    def _extract_google_metadata(self, container) -> Dict[str, Any]:
        """Extract additional metadata from Google result."""
        metadata = {}
        
        try:
            # Look for date information
            date_elements = container.find_all(['span', 'div'], string=re.compile(r'\d+ (hour|day|week|month|year)s? ago'))
            if date_elements:
                metadata['relative_date'] = date_elements[0].get_text(strip=True)
            
            # Look for site information
            cite_element = container.find('cite')
            if cite_element:
                metadata['display_url'] = cite_element.get_text(strip=True)
            
            # Look for special result types
            if container.find(class_='g-blk'):
                metadata['result_type'] = 'featured_snippet'
            elif container.find(class_='mnr-c'):
                metadata['result_type'] = 'news'
            elif container.find(class_='rlfl__tls'):
                metadata['result_type'] = 'local'
        
        except Exception as e:
            self.logger.debug(f"Error extracting Google metadata: {e}")
        
        return metadata
    
    async def search_news(
        self,
        query: str,
        max_results: int = 10,
        time_filter: str = 'week'
    ) -> SearchResponse:
        """Search Google News specifically."""
        # Add news search parameters
        news_query = f"{query} site:news.google.com OR site:bbc.com OR site:reuters.com OR site:cnn.com"
        
        return await self.search(
            query=news_query,
            max_results=max_results,
            time_filter=time_filter,
            tbm='nws'  # News search
        )
    
    async def search_recent(
        self,
        query: str,
        max_results: int = 10,
        hours: int = 24
    ) -> SearchResponse:
        """Search for recent results."""
        time_filters = {
            24: 'day',
            168: 'week',  # 24*7
            720: 'month',  # 24*30
            8760: 'year'   # 24*365
        }
        
        # Find closest time filter
        time_filter = 'day'
        for threshold, filter_name in sorted(time_filters.items()):
            if hours <= threshold:
                time_filter = filter_name
                break
        
        return await self.search(
            query=query,
            max_results=max_results,
            time_filter=time_filter
        )
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Get Google-specific statistics."""
        base_stats = self.get_stats()
        
        google_stats = {
            **base_stats,
            'engine_features': {
                'news_search': True,
                'time_filters': True,
                'location_filters': True,
                'safe_search': True
            },
            'rate_limiting': {
                'min_delay': self.min_delay,
                'max_delay': self.max_delay,
                'respectful_crawling': True
            }
        }
        
        return google_stats